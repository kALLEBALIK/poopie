# Performance Optimizer Observation Platform Inside Executable
![bench im](/assets/bm.png?raw=true)
### Usage
```zig
    var poopie: Poopie = .{
        .config = .{
            .store_result = true,
            .compare_mode = .slowest,
        },
    };

    var sampler = try poopie.createSampler(allocator, .{ .warmups = 100, .samples = 1000 });
    defer sampler.deinit();

    sampler.start();
    while (sampler.sample()) : (sampler.store()) {
        //
        // Some code to test
        //
    }
    _ = try poopie.collect(allocator, &sampler, .{ .name = "MyBench" });
```
You can overwrite poopie settings in each collection if you want
```zig
_ = try poopie.collect(allocator, &sampler, .{ .name = "MyBench", .compare_mode = .fastest });
```
or compare against a specific file
```zig
_ = try poopie.collect(allocator, &sampler, .{
    .name = "MyBench",
    .compare_mode = .file,
    .compare_file = "MyBench_10_100070223.1_1726867407409043.json",
});
```
    
Stores benchmarks in .benchmark folder if ``store_result == true`` (false by default).

#### Prev work
This shitty library is just crappy code slapped on top of Andrew's (and contributors) awesome [poop](https://github.com/andrewrk/poop) code.
