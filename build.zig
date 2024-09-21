const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("poopie", .{
        .root_source_file = b.path("src/Poopie.zig"),
        .target = target,
        .optimize = optimize,
    });

    const lib = b.addStaticLibrary(.{
        .name = "poopie",
        .root_source_file = b.path("src/Poopie.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);
}
