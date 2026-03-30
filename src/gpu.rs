use crate::neural_net::WEIGHTS_PER_SNAKE;
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
    bind_group_layout: wgpu::BindGroupLayout,
    adapter_name: String,
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
            bind_group_layout,
            adapter_name: info.name,
        })
    }

    pub async fn forward_pass(&self, snakes: &mut [Snake]) {
        let n = snakes.len();

        // Pack weights and inputs
        let mut all_weights = vec![0.0f32; n * WEIGHTS_PER_SNAKE];
        let mut all_inputs = vec![0.0f32; n * 24];

        for (s, snake) in snakes.iter().enumerate() {
            if snake.dead {
                continue;
            }
            snake
                .brain
                .pack_weights(&mut all_weights[s * WEIGHTS_PER_SNAKE..]);
            all_inputs[s * 24..(s + 1) * 24].copy_from_slice(&snake.vision);
        }

        let weight_buf = self.create_storage_buffer(&all_weights);
        let input_buf = self.create_storage_buffer(&all_inputs);

        let output_size = (n * 4 * 4) as u64; // 4 floats * 4 bytes
        let output_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nn_bg"),
            layout: &self.bind_group_layout,
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // workgroup_size is 64, so dispatch ceil(n/64) workgroups
            pass.dispatch_workgroups(((n + 63) / 64) as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &read_buf, 0, output_size);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let slice = read_buf.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = slice.get_mapped_range();
        let results: &[f32] =
            bytemuck::cast_slice(&data);

        // Apply decisions
        for (s, snake) in snakes.iter_mut().enumerate() {
            if snake.dead {
                continue;
            }
            let off = s * 4;
            snake.decision.copy_from_slice(&results[off..off + 4]);
            snake.apply_decision();
        }

        drop(data);
        read_buf.unmap();
    }

    fn create_storage_buffer(&self, data: &[f32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
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
