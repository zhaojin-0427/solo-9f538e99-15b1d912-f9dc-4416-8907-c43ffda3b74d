"""调速下载的网页流程测试：表单校验、无解提示、来源核实、原下载不变。"""
from __future__ import annotations

import io
import re

import pytest

from app import core
from app.app import create_app

VTT = """WEBVTT

1
00:00:00.000 --> 00:00:01.000
一二三四五六七八九十

2
00:00:01.000 --> 00:00:03.000
第二句字幕

3
00:00:03.000 --> 00:00:05.000
第三句
"""

CSV = """片段号,原片起毫秒,原片止毫秒
A,00:00:00.000,00:00:05.000
"""


@pytest.fixture()
def app(tmp_path):
    db = str(tmp_path / "web.db")
    application = create_app(db)
    application.config.update(TESTING=True)
    application.db_path = db
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


def _bytes(text: str):
    return io.BytesIO(text.encode("utf-8"))


def _publish_one(client) -> int:
    response = client.post(
        "/upload",
        data={
            "vtt": (_bytes(VTT), "interview.vtt"),
            "csv": (_bytes(CSV), "edit.csv"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 302
    draft_id = int(re.search(r"/draft/(\d+)", response.headers["Location"]).group(1))
    page = client.get(f"/draft/{draft_id}")
    revision = re.search(r'name="revision" value="(\d+)"', page.data.decode()).group(1)
    published = client.post(
        f"/draft/{draft_id}/publish", data={"revision": revision}
    )
    assert published.status_code == 302
    return int(re.search(r"/editions/(\d+)", published.headers["Location"]).group(1))


class TestAdjustPage:
    def test_detail_page_shows_form(self, client):
        version_no = _publish_one(client)
        page = client.get(f"/editions/{version_no}")
        html = page.data.decode()
        assert 'name="max_cps"' in html
        assert 'name="max_shift_ms"' in html
        assert f'/editions/{version_no}/adjust' in html

    def test_original_download_unchanged_after_adjustment(self, client):
        version_no = _publish_one(client)
        original = client.get(f"/editions/{version_no}/download").data
        client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        again = client.get(f"/editions/{version_no}/download").data
        assert again == original  # 原下载始终返回归档原件


class TestAdjustDownload:
    def test_valid_adjustment_downloads_vtt(self, client):
        version_no = _publish_one(client)
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("text/vtt")
        assert "attachment" in response.headers["Content-Disposition"]
        assert "adjusted" in response.headers["Content-Disposition"]
        body = response.data.decode("utf-8")
        cues = core.parse_vtt(body)
        original = core.parse_vtt(
            client.get(f"/editions/{version_no}/download").data.decode("utf-8")
        )
        # 正文、顺序、条数不变
        assert len(cues) == len(original)
        assert [c.text for c in cues] == [c.text for c in original]
        # 每条字数/秒数不超过上限，移动不超过上限，且留在片段 0-5000 内
        for cue in cues:
            chars = core.count_non_whitespace(cue.text)
            assert chars * 1000 <= 5 * (cue.end_ms - cue.start_ms)
            assert 0 <= cue.start_ms and cue.end_ms <= 5000
        # 相邻不重叠
        for earlier, later in zip(cues, cues[1:]):
            assert earlier.end_ms <= later.start_ms
        # 全版最大移动量通过响应头回传
        assert int(response.headers["X-Max-Shift-Ms"]) <= 5000

    @pytest.mark.parametrize(
        "cps,shift",
        [("0", "1000"), ("-1", "1000"), ("abc", "1000"), ("5", "-1"), ("5", "x")],
    )
    def test_invalid_params_rejected_without_file(self, client, cps, shift):
        version_no = _publish_one(client)
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": cps, "max_shift_ms": shift},
        )
        assert response.status_code == 400
        disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" not in disposition  # 不提供调整件

    def test_impossible_names_fragment_and_no_file(self, client):
        version_no = _publish_one(client)
        # 首条 10 字显示 1 秒，cps=5 需 2000 毫秒；不允许移动 => 无解
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "0"},
        )
        assert response.status_code == 422
        html = response.data.decode()
        assert "无解" in html and "A" in html
        assert "attachment" not in response.headers.get("Content-Disposition", "")

    def test_unknown_edition_404(self, client):
        response = client.post(
            "/editions/999/adjust", data={"max_cps": "5", "max_shift_ms": "1000"}
        )
        assert response.status_code == 404

    def test_tagged_cue_rejected_as_non_plain_text(self, client):
        tagged = VTT.replace("第二句字幕", "带<b>标签</b>的字幕")
        response = client.post(
            "/upload",
            data={
                "vtt": (_bytes(tagged), "tagged.vtt"),
                "csv": (_bytes(CSV), "edit.csv"),
            },
            content_type="multipart/form-data",
        )
        draft_id = int(re.search(r"/draft/(\d+)", response.headers["Location"]).group(1))
        page = client.get(f"/draft/{draft_id}")
        revision = re.search(r'name="revision" value="(\d+)"', page.data.decode()).group(1)
        published = client.post(
            f"/draft/{draft_id}/publish", data={"revision": revision}
        )
        version_no = int(re.search(r"/editions/(\d+)", published.headers["Location"]).group(1))
        result = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        assert result.status_code == 422
        assert "纯文本" in result.data.decode()
        assert "attachment" not in result.headers.get("Content-Disposition", "")


class TestSourceVerificationWeb:
    def test_replay_path_works_for_legacy_edition(self, client, app):
        version_no = _publish_one(client)
        conn = core.connect(app.db_path)
        edition = core.get_edition(conn, version_no)
        # 旧版没有逐行归档，但源草稿还在：重放一致即可调整
        conn.execute("DELETE FROM edition_cue WHERE edition_id = ?", (edition["id"],))
        conn.close()
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        assert response.status_code == 200
        assert "attachment" in response.headers["Content-Disposition"]

    def test_unverifiable_source_explains_and_no_file(self, client, app):
        version_no = _publish_one(client)
        conn = core.connect(app.db_path)
        edition = core.get_edition(conn, version_no)
        conn.execute("DELETE FROM edition_cue WHERE edition_id = ?", (edition["id"],))
        conn.execute(
            "UPDATE edition SET vtt_blob = ? WHERE id = ?",
            (edition["vtt_blob"] + b" ", edition["id"]),
        )
        conn.close()
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        assert response.status_code == 422
        html = response.data.decode()
        assert "来源无法核实" in html
        assert "attachment" not in response.headers.get("Content-Disposition", "")

    def test_adjust_still_works_after_draft_cleanup(self, client, app):
        version_no = _publish_one(client)
        conn = core.connect(app.db_path)
        draft_id = core.get_edition(conn, version_no)["source_draft_id"]
        conn.execute("DELETE FROM decision WHERE draft_id = ?", (draft_id,))
        conn.execute("DELETE FROM draft WHERE id = ?", (draft_id,))
        conn.close()
        response = client.post(
            f"/editions/{version_no}/adjust",
            data={"max_cps": "5", "max_shift_ms": "5000"},
        )
        assert response.status_code == 200
        assert "attachment" in response.headers["Content-Disposition"]
