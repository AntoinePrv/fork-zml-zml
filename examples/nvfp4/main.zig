const std = @import("std");

const zml = @import("zml");
const dquant = @import("dequant.zig");
const sfdquant = @import("safetensors.zig");

const SafeTensor = zml.safetensors.Tensor;
const TensorRegistry = zml.safetensors.TensorRegistry;
const DataType = zml.DataType;
const DEFAULT_ALIGN = dquant.DEFAULT_ALIGN;

fn dequantThread(
    registry: *const zml.safetensors.TensorRegistry,
    iter: *zml.safetensors.Tensors.Iterator,
    lock: *std.Io.RwLock,
    allocator: std.mem.Allocator,
    io: std.Io,
    id: usize,
) !void {
    var tmp = std.heap.ArenaAllocator.init(allocator);
    defer tmp.deinit();

    try lock.lock(io);
    var locked: bool = true;
    defer if (locked) lock.unlock(io);

    while (iter.next()) |t| {
        const tensor = t.value_ptr;
        const nvfp4t = try sfdquant.NvFp4SafeTensor.make(tensor, registry, tmp.allocator());
        if (nvfp4t != null) {
            // Let other thread find their work
            locked = false;
            lock.unlock(io);

            std.debug.print(
                \\  [Thread:{d}] Reading {s}
                \\      - shape: {f}
                \\      - elements: {d}
                \\
            ,
                .{ id, tensor.name, tensor.shape, nvfp4t.?.nelements() },
            );
            const buff32 = try sfdquant.tensorDequantF32(nvfp4t.?, allocator, tmp.allocator(), io);

            // Fake usage of the data
            std.mem.doNotOptimizeAway(&buff32);
            defer allocator.free(buff32);

            try lock.lock(io);
            locked = true;
        }
    }
}

// Notes:
// We could improve on the reader to avoid full allocation
// We did not account for endianness
pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const model_path = process_args[1];
    const nthreads = try std.fmt.parseInt(usize, process_args[2], 10);

    std.debug.print("Reading, {s}\n", .{model_path});

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    // Thread pool
    var threads = try std.ArrayList(std.Thread).initCapacity(allocator, 16);
    defer threads.deinit(allocator);

    // Launching threads with their
    var iter = registry.iterator();
    var lock = std.Io.RwLock.init;
    std.debug.print("Launching dequantization on {s}\n", .{nthreads});
    for (0..nthreads - 1) |_| {
        const t = try std.Thread.spawn(
            .{},
            dequantThread,
            .{ &registry, &iter, &lock, allocator, io },
        );
        try threads.append(allocator, t);
    }
    // Main thread pulls some work too
    const result = dequantThread(&registry, &iter, &lock, allocator, io);

    // Check all errors
    try result;
    for (threads.items) |t| {
        t.join();
    }
}

test {
    _ = dquant;
}
