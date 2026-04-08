FROM rust:1.88-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY static/ static/

RUN cargo build --release

FROM gcr.io/distroless/cc-debian12

WORKDIR /app
COPY --from=builder /app/target/release/snake-ai .

EXPOSE 3030
CMD ["./snake-ai"]
