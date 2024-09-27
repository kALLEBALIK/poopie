const std = @import("std");
const Allocator = std.mem.Allocator;
const File = std.fs.File;
const BufferedWriter = std.io.BufferedWriter;
const PERF = std.os.linux.PERF;
const fd_t = std.posix.fd_t;
const pid_t = std.os.pid_t;
const assert = std.debug.assert;
const progress = @import("./progress.zig");

const MAX_SAMPLES = 30000;
const PATH = ".benchmarks/";
const EXT = ".json";
const FILE_NAME_DATA_PADDING = 72;
pub const MAX_FILE_NAME_LEN = 255;
/// File name + path len + ext len + padding for file name data
const MAX_PATH_LEN = MAX_FILE_NAME_LEN + PATH.len + EXT.len + FILE_NAME_DATA_PADDING;

const Poopie = @This();

/// Keeps track of how many
/// times we printed a result
prints: usize = 0,
/// Fields can be overridden by
/// CollectConfig in sampler,
config: PoopieConfig = .{},

write_name_buf: [MAX_FILE_NAME_LEN + FILE_NAME_DATA_PADDING]u8 = undefined,
read_path_buf: [MAX_PATH_LEN]u8 = undefined,
write_name_buf_cursor: usize = 0,
read_path_buf_cursor: usize = 0,

/// Returns a sampler
pub fn createSampler(self: Poopie, allocator: Allocator, sample_config: SampleConfig) !Sampler {
    return try Sampler.init(allocator, sample_config, self.config);
}

/// Collects results from Sampler, stores and prints the results.
/// Returns slice of the filename if compare_mode == .file
/// and file_name_out_buffer != null, else returns null.
pub fn collect(self: *Poopie, allocator: Allocator, sampler: *Sampler, collect_config: CollectConfig) !?[]const u8 {
    if (!sampler.started) @panic("Sampler needs to be started before collecting it");
    if (sampler.cur_sample == 0) @panic("No samples collected");

    sampler.started = false;
    sampler.resetBar();

    var return_name_slice: ?[]const u8 = null;
    const internal_config = createInternalConfig(self.config, collect_config);

    const all_samples = sampler.samples_buf[0..sampler.sample_config.samples];
    const measurements: Measurements = .{
        .wall_time = Measurement.compute(all_samples, "wall_time", .nanoseconds),
        .max_rss = Measurement.compute(all_samples, "max_rss", .bytes),
        .cpu_cycles = Measurement.compute(all_samples, "cpu_cycles", .count),
        .instructions = Measurement.compute(all_samples, "instructions", .count),
        .cache_references = Measurement.compute(all_samples, "cache_references", .count),
        .cache_misses = Measurement.compute(all_samples, "cache_misses", .count),
        .branch_misses = Measurement.compute(all_samples, "branch_misses", .count),
    };

    var bench_count: usize = 1;
    var maybe_compare_measurement: ?std.json.Parsed(Measurements) = null;
    defer if (maybe_compare_measurement) |compare_measurement| compare_measurement.deinit();

    if (internal_config.compare_mode != .none and internal_config.print_result) {
        if (try self.readMeasurementFromFile(allocator, sampler.sample_config.samples, internal_config)) |compare_measurement| {
            bench_count += 1;
            maybe_compare_measurement = compare_measurement;
        }
    }

    if (internal_config.store_result) {
        try self.writeMeasurementsToFile(allocator, measurements, internal_config);
        if (internal_config.file_name_out_buffer) |out_buffer| {
            @memcpy(out_buffer[0..self.write_name_buf_cursor], self.write_name_buf[0..self.write_name_buf_cursor]);
            return_name_slice = out_buffer[0..self.write_name_buf_cursor];
        }
    }

    if (internal_config.print_result) {
        defer self.prints += 1;

        var stdout = std.io.getStdOut();
        var stdout_bw = std.io.bufferedWriter(stdout.writer());
        const stdout_w = stdout_bw.writer();

        if (self.prints > 0) {
            const width = progress.getScreenWidth(stdout.handle);
            for (0..width) |_| {
                try stdout_w.writeAll("─");
            }
        }

        if (maybe_compare_measurement) |compare_measurement| {
            try printMeasurementHeader(sampler.tty_conf, stdout_w, sampler.sample_config.samples, internal_config, bench_count, false);

            inline for (@typeInfo(Measurements).Struct.fields) |field| {
                const measurement = @field(compare_measurement.value, field.name);
                try printMeasurement(sampler.tty_conf, stdout_w, measurement, field.name, null, 1);
            }
            if (internal_config.compare_mode != .none) {
                try printFileName(sampler.tty_conf, stdout_w, "In: ", self.read_path_buf[PATH.len..self.read_path_buf_cursor]);
            }
            try stdout_w.writeAll("\n");
        }

        try printMeasurementHeader(sampler.tty_conf, stdout_w, sampler.sample_config.samples, internal_config, bench_count, maybe_compare_measurement != null);
        inline for (@typeInfo(Measurements).Struct.fields) |field| {
            const measurement = @field(measurements, field.name);
            const first_measurement = if (maybe_compare_measurement) |compare_measurement| blk: {
                break :blk @field(compare_measurement.value, field.name);
            } else null;
            try printMeasurement(sampler.tty_conf, stdout_w, measurement, field.name, first_measurement, bench_count);
        }
        if (internal_config.store_result) {
            try printFileName(sampler.tty_conf, stdout_w, "Out: ", self.write_name_buf[0..self.write_name_buf_cursor]);
        }
        if (internal_config.compare_mode != .none and maybe_compare_measurement == null) {
            try printFileName(sampler.tty_conf, stdout_w, "", "Comparison file not found");
        }
        try stdout_bw.flush();
    }

    return return_name_slice;
}

fn createInternalConfig(poopie_config: PoopieConfig, collect_config: CollectConfig) CollectConfigInternal {
    var internal: CollectConfigInternal = .{
        .compare_file = collect_config.compare_file,
        .name = collect_config.name,
    };
    inline for (@typeInfo(PoopieConfig).Struct.fields) |field| {
        const field_value = @field(poopie_config, field.name);
        if (@hasField(CollectConfigInternal, field.name)) {
            if (field_value) |value| {
                @field(internal, field.name) = value;
            }
        }
    }
    inline for (@typeInfo(CollectConfig).Struct.fields) |field| {
        const field_value = @field(collect_config, field.name);
        if (@hasField(CollectConfigInternal, field.name)) {
            if (@typeInfo(field.type) == .Optional and field_value != null) {
                @field(internal, field.name) = field_value.?;
            }
        }
    }
    if (internal.name.len > MAX_FILE_NAME_LEN)
        std.debug.panic("Too long name, max length is: {}", .{MAX_FILE_NAME_LEN});

    return internal;
}

fn writeMeasurementsToFile(
    self: *Poopie,
    allocator: Allocator,
    measurements: Measurements,
    config: CollectConfigInternal,
) !void {
    if (std.mem.indexOf(u8, config.name, "_") != null) @panic("Name can't contain an underscore '_'.");

    var string = std.ArrayList(u8).init(allocator);
    defer string.deinit();

    _ = try std.json.stringify(measurements, .{}, string.writer());

    std.fs.cwd().makeDir(PATH) catch |err|
        if (err != error.PathAlreadyExists) return err;

    var dir = try std.fs.cwd().openDir(PATH, .{});
    defer dir.close();

    const file_name = try std.fmt.bufPrint(&self.write_name_buf, "{s}_{d}_{d}_{d}.json", .{
        config.name,
        measurements.wall_time.sample_count,
        measurements.wall_time.mean,
        std.time.microTimestamp(),
    });
    self.write_name_buf_cursor = file_name.len;

    var file = try dir.createFile(file_name, .{});
    defer file.close();
    try file.writer().writeAll(string.items);
}

fn readMeasurementFromFile(
    self: *Poopie,
    allocator: Allocator,
    samples: usize,
    config: CollectConfigInternal,
) !?std.json.Parsed(Measurements) {
    if (config.compare_mode == .none)
        return null;

    @memcpy(self.read_path_buf[0..PATH.len], PATH[0..PATH.len]);

    switch (config.compare_mode) {
        .file => {
            if (config.compare_file) |file_name| {
                @memcpy(self.read_path_buf[PATH.len .. PATH.len + file_name.len], file_name[0..]);
                self.read_path_buf_cursor = PATH.len + file_name.len;
            } else return null;
        },
        .none => {
            return null;
        },
        else => {
            var compare: i64 = switch (config.compare_mode) {
                .oldest => std.math.maxInt(i64),
                .fastest => @bitCast(std.math.floatMax(f64)),
                else => 0,
            };

            var dir = std.fs.cwd().openDir(PATH, .{ .iterate = true }) catch |err| {
                if (err == error.FileNotFound) return null;
                return err;
            };

            var dir_iter = dir.iterate();
            while (try dir_iter.next()) |file| {
                var data: FileNameData = undefined;
                try parseFileName(file.name, &data);
                if (!std.mem.eql(u8, data.extension, EXT)) @panic("Invalid file extension");
                if (std.mem.eql(u8, data.name, config.name) and data.samples == samples) {
                    switch (config.compare_mode) {
                        .fastest => {
                            if (data.wall_time < @as(f64, @bitCast(compare))) {
                                compare = @bitCast(data.wall_time);
                                @memcpy(self.read_path_buf[PATH.len .. PATH.len + file.name.len], file.name[0..file.name.len]);
                                self.read_path_buf_cursor = file.name.len + PATH.len;
                            }
                        },
                        .slowest => {
                            if (data.wall_time > @as(f64, @bitCast(compare))) {
                                compare = @bitCast(data.wall_time);
                                @memcpy(self.read_path_buf[PATH.len .. PATH.len + file.name.len], file.name[0..file.name.len]);
                                self.read_path_buf_cursor = file.name.len + PATH.len;
                            }
                        },
                        .latest => {
                            if (data.time_stamp > compare) {
                                compare = data.time_stamp;
                                @memcpy(self.read_path_buf[PATH.len .. PATH.len + file.name.len], file.name[0..file.name.len]);
                                self.read_path_buf_cursor = file.name.len + PATH.len;
                            }
                        },
                        .oldest => {
                            if (data.time_stamp < compare) {
                                compare = data.time_stamp;
                                @memcpy(self.read_path_buf[PATH.len .. PATH.len + file.name.len], file.name[0..file.name.len]);
                                self.read_path_buf_cursor = file.name.len + PATH.len;
                            }
                        },
                        .file, .none => unreachable,
                    }
                }
            }
        },
    }

    if (self.read_path_buf_cursor == 0)
        return null;

    const file_name = self.read_path_buf[0..self.read_path_buf_cursor];
    const stat = try std.fs.cwd().statFile(file_name);
    const file_buffer = try allocator.alloc(u8, stat.size);
    defer allocator.free(file_buffer);

    var file = try std.fs.cwd().openFile(file_name, .{});
    defer file.close();
    _ = try file.readAll(file_buffer);
    return try std.json.parseFromSlice(Measurements, allocator, file_buffer, .{});
}

pub const Sampler = struct {
    poopie_config: PoopieConfig,
    timer: std.time.Timer,
    samples_buf: [MAX_SAMPLES]Sample = undefined,
    tty_conf: std.io.tty.Config,
    sample_config: SampleConfig,
    bar: progress.ProgressBar,
    started: bool,
    cur_warmup: u32,
    cur_sample: u32,
    perf_fds: [5]fd_t,
    first_start_ns: u64,
    start_ns: u64,
    start_rss: u64,

    pub fn deinit(self: *Sampler) void {
        self.bar.deinit();
    }

    fn init(allocator: Allocator, sample_config: SampleConfig, poopie_config: PoopieConfig) !Sampler {
        if (sample_config.samples > MAX_SAMPLES) std.debug.panic("Too many samples, max samples are: {}", .{MAX_SAMPLES});

        const stdout = std.io.getStdOut();
        const color: ColorMode = .auto;
        const tty_conf: std.io.tty.Config = switch (color) {
            .auto => std.io.tty.detectConfig(stdout),
            .never => .no_color,
            .ansi => .escape_codes,
        };

        return .{
            .started = false,
            .cur_warmup = 0,
            .cur_sample = 0,
            .perf_fds = [1]fd_t{-1} ** perf_measurements.len,
            .first_start_ns = 0,
            .start_ns = 0,
            .poopie_config = poopie_config,
            .tty_conf = tty_conf,
            .sample_config = sample_config,
            .bar = try progress.ProgressBar.init(allocator, stdout),
            .timer = try std.time.Timer.start(),
            .start_rss = 0,
        };
    }

    /// initiates/resets sampler, must be called before sample.
    pub fn start(self: *Sampler) void {
        if (self.started) @panic("Sampler needs to be collected before sampling it again.");
        self.cur_warmup = 0;
        self.cur_sample = 0;
        self.first_start_ns = 0;
        self.start_ns = 0;
        self.started = true;
        self.first_start_ns = self.timer.read();
        self.start_rss = 0;
    }

    /// Start collection of performance data
    pub fn sample(self: *Sampler) bool {
        if (!self.started) @panic("Sampler needs to be started before sampling it.");

        // Warmup
        if (self.cur_warmup < self.sample_config.warmups) {
            self.renderBar(progress.EscapeCodes.cyan);
            return true;
        }

        if (self.cur_sample == self.sample_config.samples)
            return false;

        self.renderBar(progress.EscapeCodes.yellow);

        for (perf_measurements, &self.perf_fds) |measurement, *perf_fd| {
            var attr: std.os.linux.perf_event_attr = .{
                .type = PERF.TYPE.HARDWARE,
                .config = @intFromEnum(measurement.config),
                .flags = .{
                    .disabled = true,
                    .exclude_kernel = true,
                    .exclude_hv = true,
                    .inherit = true,
                    .enable_on_exec = false,
                },
            };

            perf_fd.* = std.posix.perf_event_open(&attr, 0, -1, self.perf_fds[0], PERF.FLAG.FD_NO_GROUP) catch |err| {
                std.debug.panic("unable to open perf event: {s}\n", .{@errorName(err)});
            };
            _ = std.os.linux.ioctl(perf_fd.*, PERF.EVENT_IOC.ENABLE, @intFromPtr(perf_fd));
        }

        var rusage: std.os.linux.rusage = undefined;
        self.start_rss = if (std.os.linux.getrusage(std.os.linux.rusage.SELF, &rusage) == 0) @as(u64, @intCast(rusage.maxrss)) * 1024 else 0;

        _ = std.os.linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.RESET, PERF.IOC_FLAG_GROUP);
        _ = std.os.linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.ENABLE, PERF.IOC_FLAG_GROUP);

        self.start_ns = self.timer.read();

        return true;
    }

    /// Store sample
    pub fn store(self: *Sampler) void {
        const end_ns = self.timer.read() - self.start_ns;
        _ = std.os.linux.ioctl(self.perf_fds[0], PERF.EVENT_IOC.DISABLE, PERF.IOC_FLAG_GROUP);

        var rusage: std.os.linux.rusage = undefined;
        const rss_now = if (std.os.linux.getrusage(std.os.linux.rusage.SELF, &rusage) == 0) @as(u64, @intCast(rusage.maxrss)) * 1024 else 0;
        const rss = if (rss_now <= self.start_rss) 0 else rss_now - self.start_rss;

        // Warmup
        if (self.cur_warmup < self.sample_config.warmups) {
            self.updateBarEstimation(self.cur_warmup, self.sample_config.warmups, self.first_start_ns);
            if (self.cur_warmup == self.sample_config.warmups - 1) {
                self.resetBar();
                self.first_start_ns = self.timer.read();
            }
            self.cur_warmup += 1;
            return;
        }

        if (self.cur_sample == self.sample_config.samples)
            return;

        self.samples_buf[self.cur_sample] = .{
            .wall_time = end_ns,
            .max_rss = rss,
            .cpu_cycles = readPerfFd(self.perf_fds[0]),
            .instructions = readPerfFd(self.perf_fds[1]),
            .cache_references = readPerfFd(self.perf_fds[2]),
            .cache_misses = readPerfFd(self.perf_fds[3]),
            .branch_misses = readPerfFd(self.perf_fds[4]),
        };
        for (&self.perf_fds) |*perf_fd| {
            std.posix.close(perf_fd.*);
            perf_fd.* = -1;
        }

        self.updateBarEstimation(self.cur_sample, self.sample_config.samples, self.first_start_ns);
        self.cur_sample += 1;
    }

    fn resetBar(self: *Sampler) void {
        if (self.poopie_config.disable_bar)
            return;
        if (self.tty_conf != .no_color) {
            self.bar.clear() catch @panic("Couldn't clear bar");
            self.bar.current = 0;
            self.bar.estimate = 1;
        }
    }

    fn renderBar(self: *Sampler, color: []const u8) void {
        if (self.poopie_config.disable_bar)
            return;
        if (self.tty_conf != .no_color)
            self.bar.render(color) catch @panic("Couldn't render bar");
    }

    fn updateBarEstimation(self: *Sampler, cur_sample: u64, max_sample: u64, first_Start: u64) void {
        if (self.poopie_config.disable_bar)
            return;
        const max_nano_seconds: u64 = std.time.ns_per_s * 5;
        if (self.tty_conf != .no_color) {
            self.bar.estimate = est_total: {
                const cur_samples: u64 = cur_sample + 1;
                const ns_per_sample = (self.timer.read() - first_Start) / cur_samples;
                const estimate = std.math.divCeil(u64, max_nano_seconds, ns_per_sample) catch unreachable;
                break :est_total @intCast(@min(MAX_SAMPLES, @max(cur_samples, estimate, max_sample)));
            };
            self.bar.current += 1;
        }
    }
};

pub const CompareMode = enum {
    none,
    fastest,
    slowest,
    latest,
    oldest,
    file,
};

pub const SampleConfig = struct {
    warmups: u32,
    samples: u32,
};

pub const PoopieConfig = struct {
    /// Disable "loading" bar.
    disable_bar: bool = false,
    /// Store result in .benchmark folder
    store_result: ?bool = null,
    /// Print result
    print_result: ?bool = null,
    /// Compare mode, what previous
    /// benchmark should we compare against
    compare_mode: ?CompareMode = null,
};

pub const CollectConfig = struct {
    /// Store result in .benchmark folder
    store_result: ?bool = null,
    /// Print result
    print_result: ?bool = null,
    /// Compare mode, what previous
    /// benchmark should we compare against
    compare_mode: ?CompareMode = null,
    /// Specific file to compare against
    /// only used if compare_mode is .file
    compare_file: ?[]const u8 = null,
    /// out buffer for file name
    file_name_out_buffer: ?[]u8 = null,
    /// name of benchmark
    name: []const u8,
};

const CollectConfigInternal = struct {
    store_result: bool = false,
    print_result: bool = true,
    compare_mode: CompareMode = .fastest,
    compare_file: ?[]const u8 = null,
    file_name_out_buffer: ?[]u8 = null,
    name: []const u8,
};

const FileNameData = struct {
    extension: []const u8,
    name: []const u8,
    samples: u16,
    wall_time: f64,
    time_stamp: i64,
};

fn printFileName(
    tty_conf: std.io.tty.Config,
    w: anytype,
    pre: []const u8,
    name: []const u8,
) !void {
    try tty_conf.setColor(w, .dim);
    try w.print("  {s}{s}", .{ pre, name });
    try tty_conf.setColor(w, .reset);
    try w.writeAll("\n");
}

fn parseFileName(file_name: []const u8, data: *FileNameData) !void {
    if (file_name.len < EXT.len) @panic("Invalid file name.");
    const ext = file_name[file_name.len - EXT.len .. file_name.len];
    data.extension = ext;
    var token_iter = std.mem.tokenizeScalar(u8, file_name[0 .. file_name.len - EXT.len], '_');
    var curr_token: usize = 0;
    while (token_iter.next()) |token| {
        switch (curr_token) {
            0 => data.name = token,
            1 => data.samples = try std.fmt.parseInt(u16, token, 10),
            2 => data.wall_time = try std.fmt.parseFloat(f64, token),
            3 => data.time_stamp = try std.fmt.parseInt(i64, token, 10),
            else => {
                @panic("Couldn't parse file name, too many tokens.");
            },
        }
        curr_token += 1;
    }
    if (curr_token < 4)
        @panic("Couldn't parse file name, to few tokens.");
}

fn readPerfFd(fd: fd_t) usize {
    var result: usize = 0;
    const n = std.posix.read(fd, std.mem.asBytes(&result)) catch |err| {
        std.debug.panic("unable to read perf fd: {s}\n", .{@errorName(err)});
    };
    assert(n == @sizeOf(usize));
    return result;
}

const PerfMeasurement = struct {
    name: []const u8,
    config: PERF.COUNT.HW,
};

const perf_measurements = [_]PerfMeasurement{
    .{ .name = "cpu_cycles", .config = PERF.COUNT.HW.CPU_CYCLES },
    .{ .name = "instructions", .config = PERF.COUNT.HW.INSTRUCTIONS },
    .{ .name = "cache_references", .config = PERF.COUNT.HW.CACHE_REFERENCES },
    .{ .name = "cache_misses", .config = PERF.COUNT.HW.CACHE_MISSES },
    .{ .name = "branch_misses", .config = PERF.COUNT.HW.BRANCH_MISSES },
};

const Measurements = struct {
    wall_time: Measurement,
    max_rss: Measurement,
    cpu_cycles: Measurement,
    instructions: Measurement,
    cache_references: Measurement,
    cache_misses: Measurement,
    branch_misses: Measurement,
};

const Sample = struct {
    wall_time: u64,
    max_rss: u64,
    cpu_cycles: u64,
    instructions: u64,
    cache_references: u64,
    cache_misses: u64,
    branch_misses: u64,

    pub fn lessThanContext(comptime field: []const u8) type {
        return struct {
            fn lessThan(
                _: void,
                lhs: Sample,
                rhs: Sample,
            ) bool {
                return @field(lhs, field) < @field(rhs, field);
            }
        };
    }
};

const ColorMode = enum {
    auto,
    never,
    ansi,
};

const Measurement = struct {
    q1: u64,
    median: u64,
    q3: u64,
    min: u64,
    max: u64,
    mean: f64,
    std_dev: f64,
    outlier_count: u64,
    sample_count: u64,
    unit: Unit,

    const Unit = enum {
        nanoseconds,
        bytes,
        count,
    };

    fn compute(samples: []Sample, comptime field: []const u8, unit: Unit) Measurement {
        std.mem.sort(Sample, samples, {}, Sample.lessThanContext(field).lessThan);
        // Compute stats
        var total: u64 = 0;
        var min: u64 = std.math.maxInt(u64);
        var max: u64 = 0;
        for (samples) |s| {
            const v = @field(s, field);
            total += v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const mean = @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(samples.len));
        var std_dev: f64 = 0;
        for (samples) |s| {
            const v = @field(s, field);
            const delta: f64 = @as(f64, @floatFromInt(v)) - mean;
            std_dev += delta * delta;
        }
        if (samples.len > 1) {
            std_dev /= @floatFromInt(samples.len - 1);
            std_dev = @sqrt(std_dev);
        }

        const q1 = @field(samples[samples.len / 4], field);
        const q3 = if (samples.len < 4) @field(samples[samples.len - 1], field) else @field(samples[samples.len - samples.len / 4], field);
        // Tukey's Fences outliers
        var outlier_count: u64 = 0;
        const iqr: f64 = @floatFromInt(q3 - q1);
        const low_fence = @as(f64, @floatFromInt(q1)) - 1.5 * iqr;
        const high_fence = @as(f64, @floatFromInt(q3)) + 1.5 * iqr;
        for (samples) |s| {
            const v: f64 = @floatFromInt(@field(s, field));
            if (v < low_fence or v > high_fence) outlier_count += 1;
        }
        return .{
            .q1 = q1,
            .median = @field(samples[samples.len / 2], field),
            .q3 = q3,
            .mean = mean,
            .min = min,
            .max = max,
            .std_dev = std_dev,
            .outlier_count = outlier_count,
            .sample_count = samples.len,
            .unit = unit,
        };
    }
};

fn printMeasurementHeader(
    tty_conf: std.io.tty.Config,
    w: anytype,
    samples: usize,
    config: CollectConfigInternal,
    bench_count: usize,
    show_delta: bool,
) !void {
    const stdout_w = w;

    try tty_conf.setColor(stdout_w, .bold);
    try stdout_w.print("Benchmark {s}", .{config.name});
    try tty_conf.setColor(stdout_w, .dim);
    if (bench_count > 1 and !show_delta and config.compare_mode != .none)
        try stdout_w.print(" {s} run", .{@tagName(config.compare_mode)});
    if (bench_count > 1 and show_delta and config.compare_mode != .none)
        try stdout_w.print(" this run", .{});
    try stdout_w.print(" ({d} runs)", .{samples});
    try tty_conf.setColor(stdout_w, .reset);
    try stdout_w.writeAll(":");

    try stdout_w.writeAll("\n");

    try tty_conf.setColor(stdout_w, .bold);

    try stdout_w.writeAll("  measurement");
    try stdout_w.writeByteNTimes(' ', 23 - "  measurement".len);
    try tty_conf.setColor(stdout_w, .bright_green);
    try stdout_w.writeAll("mean");
    try tty_conf.setColor(stdout_w, .reset);
    try tty_conf.setColor(stdout_w, .bold);
    try stdout_w.writeAll(" ± ");
    try tty_conf.setColor(stdout_w, .green);
    try stdout_w.writeAll("σ");
    try tty_conf.setColor(stdout_w, .reset);

    try tty_conf.setColor(stdout_w, .bold);
    try stdout_w.writeByteNTimes(' ', 12);
    try tty_conf.setColor(stdout_w, .cyan);
    try stdout_w.writeAll("min");
    try tty_conf.setColor(stdout_w, .reset);
    try tty_conf.setColor(stdout_w, .bold);
    try stdout_w.writeAll(" … ");
    try tty_conf.setColor(stdout_w, .magenta);
    try stdout_w.writeAll("max");
    try tty_conf.setColor(stdout_w, .reset);

    try tty_conf.setColor(stdout_w, .bold);
    try stdout_w.writeByteNTimes(' ', 20 - " outliers".len);
    try tty_conf.setColor(stdout_w, .bright_yellow);
    try stdout_w.writeAll("outliers");
    try tty_conf.setColor(stdout_w, .reset);

    if (show_delta) {
        try tty_conf.setColor(stdout_w, .bold);
        try stdout_w.writeByteNTimes(' ', 9);
        try stdout_w.writeAll("delta");
        try tty_conf.setColor(stdout_w, .reset);
    }
    try stdout_w.writeAll("\n");
}

fn printMeasurement(
    tty_conf: std.io.tty.Config,
    w: anytype,
    m: Measurement,
    name: []const u8,
    first_m: ?Measurement,
    bench_count: usize,
) !void {
    try w.print("  {s}", .{name});

    var buf: [200]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    var count: usize = 0;

    const color_enabled = tty_conf != .no_color;
    const spaces = 32 - ("  (mean  ):".len + name.len + 2);
    try w.writeByteNTimes(' ', spaces);
    try tty_conf.setColor(w, .bright_green);
    try printUnit(fbs.writer(), m.mean, m.unit, m.std_dev, color_enabled);
    try w.writeAll(fbs.getWritten());
    count += fbs.pos;
    fbs.pos = 0;
    try tty_conf.setColor(w, .reset);
    try w.writeAll(" ± ");
    try tty_conf.setColor(w, .green);
    try printUnit(fbs.writer(), m.std_dev, m.unit, 0, color_enabled);
    try w.writeAll(fbs.getWritten());
    count += fbs.pos;
    fbs.pos = 0;
    try tty_conf.setColor(w, .reset);

    try w.writeByteNTimes(' ', 64 - ("  measurement      ".len + count + 3));
    count = 0;

    try tty_conf.setColor(w, .cyan);
    try printUnit(fbs.writer(), @floatFromInt(m.min), m.unit, m.std_dev, color_enabled);
    try w.writeAll(fbs.getWritten());
    count += fbs.pos;
    fbs.pos = 0;
    try tty_conf.setColor(w, .reset);
    try w.writeAll(" … ");
    try tty_conf.setColor(w, .magenta);
    try printUnit(fbs.writer(), @floatFromInt(m.max), m.unit, m.std_dev, color_enabled);
    try w.writeAll(fbs.getWritten());
    count += fbs.pos;
    fbs.pos = 0;
    try tty_conf.setColor(w, .reset);

    try w.writeByteNTimes(' ', 46 - (count + 1));
    count = 0;

    const outlier_percent = @as(f64, @floatFromInt(m.outlier_count)) / @as(f64, @floatFromInt(m.sample_count)) * 100;
    if (outlier_percent >= 10)
        try tty_conf.setColor(w, .yellow)
    else
        try tty_conf.setColor(w, .dim);
    try fbs.writer().print("{d: >4.0} ({d: >2.0}%)", .{ m.outlier_count, outlier_percent });
    try w.writeAll(fbs.getWritten());
    count += fbs.pos;
    fbs.pos = 0;
    try tty_conf.setColor(w, .reset);

    try w.writeByteNTimes(' ', 19 - (count + 1));

    // ratio
    if (bench_count > 1) {
        if (first_m) |f| {
            const half = blk: {
                const z = getStatScore95(m.sample_count + f.sample_count - 2);
                const n1: f64 = @floatFromInt(m.sample_count);
                const n2: f64 = @floatFromInt(f.sample_count);
                const normer = std.math.sqrt(1.0 / n1 + 1.0 / n2);
                const numer1 = (n1 - 1) * (m.std_dev * m.std_dev);
                const numer2 = (n2 - 1) * (f.std_dev * f.std_dev);
                const df = n1 + n2 - 2;
                const sp = std.math.sqrt((numer1 + numer2) / df);
                break :blk (z * sp * normer) * 100 / f.mean;
            };
            const diff_mean_percent = (m.mean - f.mean) * 100 / f.mean;
            // significant only if full interval is beyond abs 1% with the same sign
            const is_sig = blk: {
                if (diff_mean_percent >= 1 and (diff_mean_percent - half) >= 1) {
                    break :blk true;
                } else if (diff_mean_percent <= -1 and (diff_mean_percent + half) <= -1) {
                    break :blk true;
                } else {
                    break :blk false;
                }
            };
            if (m.mean > f.mean) {
                if (is_sig) {
                    try w.writeAll("💩");
                    try tty_conf.setColor(w, .bright_red);
                } else {
                    try tty_conf.setColor(w, .dim);
                    try w.writeAll("  ");
                }
                try w.writeAll("+");
            } else {
                if (is_sig) {
                    try tty_conf.setColor(w, .bright_yellow);
                    try w.writeAll("⚡");
                    try tty_conf.setColor(w, .bright_green);
                } else {
                    try tty_conf.setColor(w, .dim);
                    try w.writeAll("  ");
                }
                try w.writeAll("-");
            }
            try fbs.writer().print("{d: >5.1}% ± {d: >4.1}%", .{ @abs(diff_mean_percent), half });
            try w.writeAll(fbs.getWritten());
            count += fbs.pos;
            fbs.pos = 0;
        } else {
            try tty_conf.setColor(w, .dim);
            try w.writeAll("0%");
        }
    }

    try tty_conf.setColor(w, .reset);
    try w.writeAll("\n");
}

fn printNum3SigFigs(w: anytype, num: f64) !void {
    if (num >= 1000 or @round(num) == num) {
        try w.print("{d: >4.0}", .{num});
        // TODO Do we need special handling here since it overruns 3 sig figs?
    } else if (num >= 100) {
        try w.print("{d: >4.0}", .{num});
    } else if (num >= 10) {
        try w.print("{d: >3.1}", .{num});
    } else {
        try w.print("{d: >3.2}", .{num});
    }
}

fn printUnit(w: anytype, x: f64, unit: Measurement.Unit, std_dev: f64, color_enabled: bool) !void {
    _ = std_dev; // TODO something useful with this
    const num = x;
    var val: f64 = 0;
    const color: []const u8 = progress.EscapeCodes.dim ++ progress.EscapeCodes.white;
    var ustr: []const u8 = "  ";
    if (num >= 1000_000_000_000) {
        val = num / 1000_000_000_000;
        ustr = switch (unit) {
            .count => "T ",
            .nanoseconds => "ks",
            .bytes => "TB",
        };
    } else if (num >= 1000_000_000) {
        val = num / 1000_000_000;
        ustr = switch (unit) {
            .count => "G ",
            .nanoseconds => "s ",
            .bytes => "GB",
        };
    } else if (num >= 1000_000) {
        val = num / 1000_000;
        ustr = switch (unit) {
            .count => "M ",
            .nanoseconds => "ms",
            .bytes => "MB",
        };
    } else if (num >= 1000) {
        val = num / 1000;
        ustr = switch (unit) {
            .count => "K ",
            .nanoseconds => "us",
            .bytes => "KB",
        };
    } else {
        val = num;
        ustr = switch (unit) {
            .count => "  ",
            .nanoseconds => "ns",
            .bytes => "  ",
        };
    }
    try printNum3SigFigs(w, val);
    if (color_enabled) {
        try w.print("{s}{s}{s}", .{ color, ustr, progress.EscapeCodes.reset });
    } else {
        try w.writeAll(ustr);
    }
}

// Gets either the T or Z score for 95% confidence.
// If no `df` variable is provided, Z score is provided.
pub fn getStatScore95(df: ?u64) f64 {
    if (df) |dff| {
        const dfv: usize = @intCast(dff);
        if (dfv <= 30) {
            return t_table95_1to30[dfv - 1];
        } else if (dfv <= 120) {
            const idx_10s = @divFloor(dfv, 10);
            return t_table95_10s_10to120[idx_10s - 1];
        }
    }
    return 1.96;
}

const t_table95_1to30 = [_]f64{
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.16,
    2.145,
    2.131,
    2.12,
    2.11,
    2.101,
    2.093,
    2.086,
    2.08,
    2.074,
    2.069,
    2.064,
    2.06,
    2.056,
    2.052,
    2.045,
    2.048,
    2.042,
};

const t_table95_10s_10to120 = [_]f64{
    2.228,
    2.086,
    2.042,
    2.021,
    2.009,
    2,
    1.994,
    1.99,
    1.987,
    1.984,
    1.982,
    1.98,
};
