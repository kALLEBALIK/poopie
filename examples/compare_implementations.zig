const std = @import("std");
const Poopie = @import("poopie");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    const allocator = gpa.allocator();

    var poopie: Poopie = .{
        .config = .{
            .store_result = true,
        },
    };

    var name_buffer: [Poopie.MAX_FILE_NAME_LEN]u8 = undefined;
    var sampler = try poopie.createSampler(allocator, .{ .warmups = 100, .samples = 500 });
    defer sampler.deinit();

    sampler.start();
    while (sampler.sample()) : (sampler.store()) {
        impl1();
    }
    const name_slice = try poopie.collect(allocator, &sampler, .{
        .name = "Bench impl1",
        .file_name_out_buffer = &name_buffer,
    });

    sampler.start();
    while (sampler.sample()) : (sampler.store()) {
        impl2();
    }
    _ = try poopie.collect(allocator, &sampler, .{
        .name = "Bench impl2",
        .compare_mode = .file,
        .compare_file = name_slice,
    });
}

fn impl1() !void {
    // testing stuff
}

fn impl2() !void {
    // testing stuff
}
