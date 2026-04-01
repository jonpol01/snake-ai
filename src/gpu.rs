use crate::neural_net::WEIGHTS_PER_SNAKE;
use crate::population::POP_SIZE;
use crate::snake::Snake;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outputs: array<f32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let snakeIdx = gid.x;
    if (snakeIdx >= arrayLength(&outputs) / 4u) {
        return;
    }

    let wBase = snakeIdx * 740u;
    let iBase = snakeIdx * 24u;
    let oBase = snakeIdx * 4u;

    // Layer 0: input(24) + bias -> hidden(16), weight matrix 16x25
    var hidden0: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        var sum: f32 = 0.0;
        for (var j = 0u; j < 24u; j++) {
            sum += weights[wBase + i * 25u + j] * inputs[iBase + j];
        }
        sum += weights[wBase + i * 25u + 24u]; // bias
        hidden0[i] = sigmoid(sum);
    }

    // Layer 1: hidden(16) + bias -> hidden(16), weight matrix 16x17
    let l1Base = wBase + 400u;
    var hidden1: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        var sum: f32 = 0.0;
        for (var j = 0u; j < 16u; j++) {
            sum += weights[l1Base + i * 17u + j] * hidden0[j];
        }
        sum += weights[l1Base + i * 17u + 16u]; // bias
        hidden1[i] = sigmoid(sum);
    }

    // Layer 2: hidden(16) + bias -> output(4), weight matrix 4x17
    let l2Base = wBase + 672u;
    for (var i = 0u; i < 4u; i++) {
        var sum: f32 = 0.0;
        for (var j = 0u; j < 16u; j++) {
            sum += weights[l2Base + i * 17u + j] * hidden1[j];
        }
        sum += weights[l2Base + i * 17u + 16u]; // bias
        outputs[oBase + i] = sigmoid(sum);
    }
}
"#;

pub struct GpuCompute {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    // Persistent buffers — allocated once, reused every step
    weight_buf: wgpu::Buffer,
    input_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl GpuCompute {
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .ok()?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("snake_nn"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nn_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nn_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nn_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pre-allocate persistent buffers for POP_SIZE snakes
        let weight_size = (POP_SIZE * WEIGHTS_PER_SNAKE * 4) as u64;
        let input_size = (POP_SIZE * 24 * 4) as u64;
        let output_size = (POP_SIZE * 4 * 4) as u64;

        let weight_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weights"),
            size: weight_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let input_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("inputs"),
            size: input_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: weight_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let info = adapter.get_info();
        eprintln!("GPU initialized: {}", info.name);
        eprintln!("  Backend: {:?}", info.backend);
        eprintln!("  Device type: {:?}", info.device_type);
        eprintln!("  Driver: {}", info.driver);
        eprintln!("  Driver info: {}", info.driver_info);

        Some(Self {
            device,
            queue,
            pipeline,
            adapter_name: info.name,
            weight_buf,
            input_buf,
            output_buf,
            read_buf,
            bind_group,
        })
    }

    /// Upload all snake weights to GPU. Call once per generation (after natural_selection).
    pub fn upload_weights(&self, snakes: &[Snake]) {
        let n = snakes.len();
        let mut all_weights = vec![0.0f32; n * WEIGHTS_PER_SNAKE];
        for (s, snake) in snakes.iter().enumerate() {
            snake.brain.pack_weights(&mut all_weights[s * WEIGHTS_PER_SNAKE..]);
        }
        self.queue.write_buffer(&self.weight_buf, 0, bytemuck::cast_slice(&all_weights));
    }

    /// Forward pass — only uploads vision inputs (weights already on GPU).
    pub async fn forward_pass(&self, snakes: &mut [Snake]) {
        let n = snakes.len();

        // Pack only inputs (weights are already on GPU from upload_weights)
        let mut all_inputs = vec![0.0f32; n * 24];
        for (s, snake) in snakes.iter().enumerate() {
            if snake.dead {
                continue;
            }
            all_inputs[s * 24..(s + 1) * 24].copy_from_slice(&snake.vision);
        }

        // Write inputs to existing buffer (no allocation)
        self.queue.write_buffer(&self.input_buf, 0, bytemuck::cast_slice(&all_inputs));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(((n + 63) / 64) as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&self.output_buf, 0, &self.read_buf, 0, (n * 4 * 4) as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let slice = self.read_buf.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let results: &[f32] = bytemuck::cast_slice(&data);

        for (s, snake) in snakes.iter_mut().enumerate() {
            if snake.dead {
                continue;
            }
            let off = s * 4;
            snake.decision.copy_from_slice(&results[off..off + 4]);
            snake.apply_decision();
        }

        drop(data);
        self.read_buf.unmap();
    }
}

/// CPU fallback forward pass for all alive snakes
pub fn cpu_forward_pass(snakes: &mut [Snake]) {
    for snake in snakes.iter_mut() {
        if snake.dead {
            continue;
        }
        snake.decision = snake.brain.forward(&snake.vision);
        snake.apply_decision();
    }
}
