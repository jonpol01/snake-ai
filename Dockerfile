FROM rust:1.88-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY static/ static/
COPY src-tauri/Cargo.toml src-tauri/Cargo.toml
COPY src-tauri/build.rs src-tauri/build.rs
COPY src-tauri/src/ src-tauri/src/
COPY src-tauri/tauri.conf.json src-tauri/tauri.conf.json
COPY src-tauri/capabilities/ src-tauri/capabilities/
COPY src-tauri/icons/32x32.png src-tauri/icons/32x32.png
COPY src-tauri/icons/128x128.png src-tauri/icons/128x128.png
COPY src-tauri/icons/128x128@2x.png src-tauri/icons/128x128@2x.png
COPY src-tauri/icons/icon.icns src-tauri/icons/icon.icns
COPY src-tauri/icons/icon.ico src-tauri/icons/icon.ico

# Only build the headless binary (not the Tauri desktop app)
RUN cargo build --release -p snake-ai

FROM gcr.io/distroless/cc-debian12

WORKDIR /app
COPY --from=builder /app/target/release/snake-ai .

EXPOSE 3030
CMD ["./snake-ai"]
