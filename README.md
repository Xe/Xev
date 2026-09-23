# xev

This service scores decision options with a local GGUF model. It follows the
[NobodyWho Jev example](https://www.nobodywho.ai/posts/jev-in-25-lines/).
The example is a parody, and its token scores are not calibrated probabilities.

The gRPC API is `xeiaso.net.xev.v1.DecisionService/Pick`. The response maps use
each option string as a key. Therefore, option strings must be unique.
`MODEL_UNSPECIFIED` selects `MODEL_QWEN3_600M`. The service accepts 2 to 26 options.

## Run locally

1. Install Python 3.12, [uv](https://docs.astral.sh/uv/), and [Buf](https://buf.build/docs/installation/).
2. Run `uv sync --locked`.
3. Run `uv run python -m decision_service.server`.

The first request downloads `Qwen/Qwen3-0.6B-GGUF` from Hugging Face. Set
`HF_HOME` to choose the model cache directory. Set `PORT` to change the gRPC port.

To regenerate the Python bindings, run `buf generate`. The schema is in
`decision_service/decision.proto`. Run `buf lint` to check the schema.

## Docker

Build and run the container:

```sh
docker build -t xev .
docker run --rm -p 50051:50051 -v xev-models:/app/.cache/huggingface xev
```

The volume keeps the model after the container stops. The image downloads the
model on the first request. The gRPC port uses plaintext; place the service
behind a TLS proxy when clients connect over an untrusted network.

## Internal deployment

[`app.yaml`](app.yaml) creates an internal Service on port `50051` with a gRPC
health check. It mounts a 2 GiB PVC at the Hugging Face model cache path.
The deployment uses one replica because the PVC uses `ReadWriteOnce` access.
It does not enable public ingress. The GitHub Actions workflow publishes
`ghcr.io/xe/xev:latest` and a commit SHA tag after tests pass on `main`.
It builds `linux/amd64` and `linux/arm64` on separate native runners, then
combines their images into one manifest.
If the GHCR package is private, give the cluster a pull secret before you apply
the manifest. You can also change `spec.image` to an image that the cluster can
pull.

The server reports `SERVING` for the empty service name and
`xeiaso.net.xev.v1.DecisionService`. During shutdown, it reports
`NOT_SERVING`. The health check confirms that the gRPC server accepts requests;
the first `Pick` request can still take time to download and load the model.

## API

Send a `PickRequest` with `context`, `question`, and `options`. The response
contains raw `logits`, normalized `log_probabilities`, and `confidence` for the
options in the request. `execution_time` includes model load time on the first
request.

The model scores one token per option: `A`, `B`, and so on. Confidence values
sum to one across the supplied options. They are model scores, not measured
accuracy or calibrated confidence.

For a command-line request, use `grpcurl` from the project root:

```sh
grpcurl -plaintext -import-path . -proto decision_service/decision.proto \
  -d '{"context":"Payroll asks for your password on a non-company sign-in page.","question":"What kind of email is this?","options":["Legitimate","Spam","Phishing"]}' \
  localhost:50051 xeiaso.net.xev.v1.DecisionService/Pick
```
