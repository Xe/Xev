"""gRPC server for the local decision model."""

from __future__ import annotations

import os
import time
from concurrent import futures

import grpc
from google.protobuf.duration_pb2 import Duration
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from decision_service import decision_pb2, decision_pb2_grpc
from decision_service.classifier import Classifier, LABELS

SERVICE_NAME = decision_pb2.DESCRIPTOR.services_by_name["DecisionService"].full_name


class DecisionService(decision_pb2_grpc.DecisionServiceServicer):
    def __init__(self, classifier=None):
        self.classifier = classifier or Classifier()

    def Pick(self, request, context):
        if request.model not in (
            decision_pb2.MODEL_UNSPECIFIED,
            decision_pb2.MODEL_QWEN3_600M,
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Unsupported model")
        if not request.question.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "question is required")
        options = list(request.options)
        if not 2 <= len(options) <= len(LABELS):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Provide 2 to 26 options")
        if any(not option.strip() for option in options) or len(set(options)) != len(options):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Options must be nonempty and unique")

        start = time.perf_counter()
        try:
            logits, log_probabilities, confidence = self.classifier.pick(
                request.context, request.question, options
            )
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        elapsed = time.perf_counter() - start
        duration = Duration()
        duration.FromNanoseconds(round(elapsed * 1_000_000_000))
        return decision_pb2.PickResponse(
            execution_time=duration,
            logits=logits,
            log_probabilities=log_probabilities,
            confidence=confidence,
        )


def create_server(classifier=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    decision_pb2_grpc.add_DecisionServiceServicer_to_server(
        DecisionService(classifier), server
    )
    health_service = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    health_service.set("", health_pb2.HealthCheckResponse.SERVING)
    health_service.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    return server, health_service


def serve():
    port = int(os.environ.get("PORT", "50051"))
    server, health_service = create_server()
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"DecisionService listening on {port}", flush=True)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        health_service.enter_graceful_shutdown()
        server.stop(grace=5).wait()


if __name__ == "__main__":
    serve()
