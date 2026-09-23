variable "GIT_SHA" {
  default = "dev"
}

target "image" {
  context    = "."
  dockerfile = "Dockerfile"
  platforms  = ["linux/amd64", "linux/arm64"]
  tags = [
    "ghcr.io/xe/xev:latest",
    "ghcr.io/xe/xev:sha-${GIT_SHA}",
  ]
  labels = {
    "org.opencontainers.image.source" = "https://github.com/Xe/Xev"
  }
}
