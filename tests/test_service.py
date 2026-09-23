from concurrent import futures

import grpc
import pytest

from decision_service import decision_pb2, decision_pb2_grpc
from decision_service.server import DecisionService


class FakeClassifier:
    def pick(self, context, question, options):
        assert context == "Payroll email"
        assert question == "What kind of message is this?"
        assert options == ["Legitimate", "Phishing"]
        return (
            {"Legitimate": 1.0, "Phishing": 2.0},
            {"Legitimate": -1.31, "Phishing": -0.31},
            {"Legitimate": 0.27, "Phishing": 0.73},
        )


@pytest.fixture
def stub():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    decision_pb2_grpc.add_DecisionServiceServicer_to_server(
        DecisionService(FakeClassifier()), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        yield decision_pb2_grpc.DecisionServiceStub(channel)
    server.stop(grace=0).wait()


def test_pick_over_grpc(stub):
    response = stub.Pick(
        decision_pb2.PickRequest(
            context="Payroll email",
            question="What kind of message is this?",
            options=["Legitimate", "Phishing"],
            model=decision_pb2.MODEL_QWEN3_600M,
        )
    )
    assert response.confidence["Phishing"] == pytest.approx(0.73)
    assert response.logits["Legitimate"] == pytest.approx(1.0)
    assert response.execution_time.ToNanoseconds() >= 0


@pytest.mark.parametrize(
    "pick_request",
    [
        decision_pb2.PickRequest(question="", options=["A", "B"]),
        decision_pb2.PickRequest(question="Pick", options=["A"]),
        decision_pb2.PickRequest(question="Pick", options=["A", "A"]),
        decision_pb2.PickRequest(question="Pick", options=["A", "B"], model=99),
    ],
)
def test_invalid_request(stub, pick_request):
    with pytest.raises(grpc.RpcError) as exc:
        stub.Pick(pick_request)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
