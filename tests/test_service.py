import grpc
import pytest
from gnostic.openapi.v3 import annotations_pb2 as openapi_annotations
from google.api import annotations_pb2 as http_annotations
from grpc_health.v1 import health_pb2, health_pb2_grpc

from decision_service import decision_pb2, decision_pb2_grpc
from decision_service.server import SERVICE_NAME, create_server


class FakeClassifier:
    def pick(self, context, question, options):
        assert context == "Payroll email"
        assert question == "What kind of message is this?"
        assert len(options) == 2
        return (
            {options[0]: 1.0, options[1]: 2.0},
            {options[0]: -1.31, options[1]: -0.31},
            {options[0]: 0.27, options[1]: 0.73},
        )


@pytest.fixture
def server_channel():
    server, health_service = create_server(FakeClassifier())
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        yield channel, health_service
    server.stop(grace=0).wait()


@pytest.fixture
def stub(server_channel):
    channel, _ = server_channel
    return decision_pb2_grpc.DecisionServiceStub(channel)


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
        decision_pb2.PickRequest(question="Pick", options=["A", "B"]),
        decision_pb2.PickRequest(question="", options=["A", "B"]),
        decision_pb2.PickRequest(question="  \n", options=["A", "B"]),
        decision_pb2.PickRequest(question="Pick", options=["A"]),
        decision_pb2.PickRequest(question="Pick", options=["A", "A"]),
        decision_pb2.PickRequest(question="Pick", options=["A", " \t"]),
        decision_pb2.PickRequest(question="Pick", options=[str(i) for i in range(27)]),
        decision_pb2.PickRequest(question="Pick", options=["A", "B"], model=99),
    ],
)
def test_invalid_request(stub, pick_request):
    with pytest.raises(grpc.RpcError) as exc:
        stub.Pick(pick_request)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_noul_uses_yes_and_no(stub):
    response = stub.Noul(
        decision_pb2.NoulRequest(
            context="Payroll email",
            question="What kind of message is this?",
        )
    )
    assert isinstance(response, decision_pb2.NoulResponse)
    assert response.confidence["yes"] == pytest.approx(0.27)
    assert response.confidence["no"] == pytest.approx(0.73)


def test_http_and_openapi_annotations():
    service = decision_pb2.DESCRIPTOR.services_by_name["DecisionService"]
    document = decision_pb2.DESCRIPTOR.GetOptions().Extensions[
        openapi_annotations.document
    ]
    assert document.info.title == "Xev Decision API"
    assert document.info.version == "v1"
    for method_name, route in (
        ("Pick", "/v1/xev/decision"),
        ("Noul", "/v1/xev/noul"),
    ):
        options = service.methods_by_name[method_name].GetOptions()
        assert options.Extensions[http_annotations.http].post == route
        assert options.Extensions[http_annotations.http].body == "*"
        assert options.Extensions[openapi_annotations.operation].summary


@pytest.mark.parametrize(
    "noul_request",
    [
        decision_pb2.NoulRequest(question="Pick"),
        decision_pb2.NoulRequest(question=""),
        decision_pb2.NoulRequest(question=" \n "),
        decision_pb2.NoulRequest(question="Pick", model=99),
    ],
)
def test_invalid_noul_request(stub, noul_request):
    with pytest.raises(grpc.RpcError) as exc:
        stub.Noul(noul_request)
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_grpc_health_check(server_channel):
    channel, health_service = server_channel
    stub = health_pb2_grpc.HealthStub(channel)
    for service in ("", SERVICE_NAME):
        response = stub.Check(health_pb2.HealthCheckRequest(service=service))
        assert response.status == health_pb2.HealthCheckResponse.SERVING

    with pytest.raises(grpc.RpcError) as exc:
        stub.Check(health_pb2.HealthCheckRequest(service="unknown"))
    assert exc.value.code() == grpc.StatusCode.NOT_FOUND

    health_service.enter_graceful_shutdown()
    response = stub.Check(health_pb2.HealthCheckRequest(service=""))
    assert response.status == health_pb2.HealthCheckResponse.NOT_SERVING
