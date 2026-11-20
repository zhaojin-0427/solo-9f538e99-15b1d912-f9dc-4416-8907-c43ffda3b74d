"""调速引擎与来源核实的单元测试。"""
from __future__ import annotations

import pytest

from app import core


def acue(start, end, text, frag="A", fstart=0, fend=10_000):
    return core.AdjustCue(
        start_ms=start,
        end_ms=end,
        text=text,
        fragment_no=frag,
        fragment_excerpt_start=fstart,
        fragment_excerpt_end=fend,
    )


class TestCharCount:
    def test_non_whitespace_only(self):
        assert core.count_non_whitespace("一句话") == 3
        assert core.count_non_whitespace(" 一 句\n话\t") == 3
        # 多行字幕的换行不计字数
        assert core.count_non_whitespace("第一行\n第二行") == 6
        # 全角空白也不计
        assert core.count_non_whitespace("甲　乙") == 2


class TestPlainText:
    def test_plain_text_passes(self):
        cues = [core.Cue(0, "", 0, 1000, "纯文本", "")]
        core.ensure_plain_text(cues)  # 不抛异常即可

    def test_tag_rejected(self):
        cues = [core.Cue(0, "", 0, 1000, "带<b>标签</b>", "")]
        with pytest.raises(core.NotPlainText):
            core.ensure_plain_text(cues)

    def test_class_rejected(self):
        cues = [core.Cue(0, "", 0, 1000, "纯文本", "align:start")]
        with pytest.raises(core.NotPlainText):
            core.ensure_plain_text(cues)


class TestAdjustEngine:
    def test_extends_short_cue_to_meet_cps(self):
        # 10 个字、显示 1 秒；上限 5 字/秒 -> 至少 2000 毫秒
        result = core.adjust_cues([acue(0, 1000, "一二三四五六七八九十")], 5, 5000)
        assert len(result) == 1
        start, end = result[0].start_ms, result[0].end_ms
        assert end - start == 2000
        assert 10 * 1000 / (end - start) <= 5
        # 起点可不动，终点后移 1000
        assert start == 0 and end == 2000
        assert result[0].max_shift_ms == 1000

    def test_ceil_duration_for_non_divisible_chars(self):
        # 7 字 / 3 字每秒 -> ceil(7000/3) = 2334 毫秒
        result = core.adjust_cues([acue(0, 1000, "一二三四五六七")], 3, 9000)
        assert result[0].end_ms - result[0].start_ms == 2334

    def test_already_slow_enough_is_unchanged(self):
        cues = [acue(1000, 5000, "三个字")]  # 3 字 / 4 秒，远低于上限
        result = core.adjust_cues(cues, 5, 0)
        assert result[0].start_ms == 1000
        assert result[0].end_ms == 5000
        assert result[0].max_shift_ms == 0

    def test_stays_inside_fragment_bounds(self):
        # 片段只到 1500，字幕 0-1000 需延长到 2000，最多只能到 1500 => 无解
        with pytest.raises(core.AdjustmentImpossible) as exc:
            core.adjust_cues(
                [acue(0, 1000, "一二三四五六七八九十", fend=1500)], 5, 5000
            )
        assert exc.value.fragment_nos == ["A"]

    def test_cannot_exceed_max_shift(self):
        # 需要终点后移 1000，但只允许移动 500 => 无解
        with pytest.raises(core.AdjustmentImpossible):
            core.adjust_cues([acue(0, 1000, "一二三四五六七八九十")], 5, 500)

    def test_minimizes_max_movement_across_whole_edition(self):
        # 两条字幕：第一条刚好达标，第二条必须延长 1000 毫秒
        cues = [
            acue(0, 2000, "一二三四五六七八九十", frag="A", fend=20_000),
            acue(2000, 3000, "一二三四五六七八九十", frag="A", fend=20_000),
        ]
        result = core.adjust_cues(cues, 5, 9000)
        shifts = [cue.max_shift_ms for cue in result]
        # 全版最大移动量应被压到最小：只需 1000
        assert max(shifts) == 1000

    def test_adjacent_cues_touch_but_never_overlap(self):
        cues = [
            acue(0, 1000, "一二三四五六", frag="A", fend=10_000),    # 需 1200
            acue(1000, 2000, "一二三四五六", frag="A", fend=10_000),  # 需 1200
        ]
        result = core.adjust_cues(cues, 5, 9000)
        first, second = result
        assert first.end_ms <= second.start_ms
        assert first.end_ms == second.start_ms  # 可以相接
        assert second.end_ms <= 10_000

    def test_cues_in_different_fragments_each_within_own_bounds(self):
        cues = [
            acue(0, 1000, "一二三四五六", frag="A", fstart=0, fend=2000),
            acue(2000, 3000, "一二三四五六", frag="B", fstart=2000, fend=4000),
        ]
        result = core.adjust_cues(cues, 5, 9000)
        assert 0 <= result[0].start_ms and result[0].end_ms <= 2000
        assert 2000 <= result[1].start_ms and result[1].end_ms <= 4000

    def test_impossible_names_the_offending_fragment(self):
        cues = [
            acue(0, 1000, "一字", frag="A", fend=5000),
            acue(0, 1000, "一二三四五六七八九十", frag="B", fstart=0, fend=1500),
        ]
        with pytest.raises(core.AdjustmentImpossible) as exc:
            core.adjust_cues(cues, 5, 9000)
        assert exc.value.fragment_nos == ["B"]

    def test_text_order_and_count_preserved(self):
        cues = [
            acue(0, 1000, "甲", frag="A", fend=9000),
            acue(1000, 2000, "乙\n两行", frag="A", fend=9000),
        ]
        result = core.adjust_cues(cues, 1, 9000)
        assert [cue.text for cue in result] == ["甲", "乙\n两行"]
        assert len(result) == 2
        rebuilt = core.build_adjusted_vtt(result)
        reparsed = core.parse_vtt(rebuilt)
        assert [cue.text for cue in reparsed] == ["甲", "乙\n两行"]
        assert rebuilt.endswith("\n")

    def test_all_endpoints_integer_milliseconds(self):
        result = core.adjust_cues([acue(333, 999, "一二三四五六七")], 3, 9000)
        assert isinstance(result[0].start_ms, int)
        assert isinstance(result[0].end_ms, int)

    def test_empty_input(self):
        assert core.adjust_cues([], 5, 100) == []

    def test_rate_constraint_holds_for_every_cue(self):
        cues = [
            acue(0, 500, "一二三四五", frag="A", fend=20_000),
            acue(500, 800, "一二三四五六七八", frag="A", fend=20_000),
            acue(800, 1200, "一二三", frag="A", fend=20_000),
        ]
        result = core.adjust_cues(cues, 4, 15_000)
        for cue in result:
            chars = core.count_non_whitespace(cue.text)
            assert chars * 1000 <= 4 * (cue.end_ms - cue.start_ms)


class TestAdjustmentArchive:
    VTT = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:05.000\n整条在A\n\n"
        "00:00:04.000 --> 00:00:07.000\n被剪口截断\n\n"
        "00:00:07.000 --> 00:00:09.000\n整条在B\n"
    )

    def _conn(self, tmp_path, name):
        path = str(tmp_path / name)
        core.init_db(path)
        return core.connect(path)

    def _publish(self, conn):
        cuts = [core.Cut("A", 1000, 5000, 2), core.Cut("B", 5000, 9000, 3)]
        cues = [
            core.Cue(0, "", 1000, 5000, "整条在A"),
            core.Cue(1, "", 4000, 7000, "被剪口截断"),
            core.Cue(2, "", 7000, 9000, "整条在B"),
        ]
        draft_id = core.create_draft(conn, self.VTT, cuts, cues)
        bundle = core.load_bundle(conn, draft_id)
        cut_occs = [o for o in bundle["occurrences"] if o.kind == core.CUT]
        for occ in cut_occs:
            core.save_decisions(
                conn,
                draft_id,
                bundle["row"]["revision"],
                [{"signature": occ.signature, "status": "keep", "text": "保留文字"}],
            )
        # 保存会推进版本锁，需重新取最新 revision 再发布。
        bundle = core.load_bundle(conn, draft_id)
        version_no = core.publish(conn, draft_id, bundle["row"]["revision"])
        return version_no, draft_id

    def test_publish_archives_every_output_cue(self, tmp_path):
        conn = self._conn(tmp_path, "a.db")
        version_no, _ = self._publish(conn)
        edition = core.get_edition(conn, version_no)
        rows = core.edition_cue_rows(conn, edition["id"])
        # 被剪口截断的字幕在 A、B 各出现一次 => 4 条成品
        assert len(rows) == edition["cue_count"] == 4
        assert [r["cue_order"] for r in rows] == [0, 1, 2, 3]
        # 片段号与片段节选起止都已归档
        assert [r["fragment_no"] for r in rows] == ["A", "A", "B", "B"]
        assert rows[0]["fragment_excerpt_start"] == 0
        assert rows[0]["fragment_excerpt_end"] == 4000  # A 时长 4000
        assert rows[2]["fragment_excerpt_start"] == 4000  # B 从 4000 开始
        assert rows[2]["fragment_excerpt_end"] == 8000
        # 归档时码/正文与原件逐条一致
        blob_cues = core.parse_vtt(edition["vtt_blob"].decode())
        for row, cue in zip(rows, blob_cues):
            assert row["start_ms"] == cue.start_ms
            assert row["end_ms"] == cue.end_ms
            assert row["text"] == cue.text
        conn.close()

    def test_adjust_works_after_source_draft_deleted(self, tmp_path):
        conn = self._conn(tmp_path, "b.db")
        version_no, draft_id = self._publish(conn)
        conn.execute("DELETE FROM decision WHERE draft_id = ?", (draft_id,))
        conn.execute("DELETE FROM draft WHERE id = ?", (draft_id,))
        adjusted = core.prepare_adjustment(conn, version_no, 5, 9000)
        assert adjusted is not None and len(adjusted) == 4
        rebuilt = core.build_adjusted_vtt(adjusted)
        assert core.parse_vtt(rebuilt)  # 仍是合法 WebVTT
        conn.close()

    def test_old_edition_replay_path_matches_blob(self, tmp_path):
        conn = self._conn(tmp_path, "c.db")
        version_no, _ = self._publish(conn)
        edition = core.get_edition(conn, version_no)
        # 模拟旧版：删除逐行归档，只留原件与源草稿
        conn.execute("DELETE FROM edition_cue WHERE edition_id = ?", (edition["id"],))
        adjusted = core.prepare_adjustment(conn, version_no, 5, 9000)
        assert len(adjusted) == 4
        conn.close()

    def test_old_edition_replay_mismatch_rejected(self, tmp_path):
        conn = self._conn(tmp_path, "d.db")
        version_no, _ = self._publish(conn)
        edition = core.get_edition(conn, version_no)
        conn.execute("DELETE FROM edition_cue WHERE edition_id = ?", (edition["id"],))
        # 篡改归档原件 => 重放结果无法逐字节一致
        conn.execute(
            "UPDATE edition SET vtt_blob = ? WHERE id = ?",
            (edition["vtt_blob"] + b" ", edition["id"]),
        )
        with pytest.raises(core.SourceUnverified, match="逐字节"):
            core.prepare_adjustment(conn, version_no, 5, 9000)
        conn.close()

    def test_old_edition_without_draft_rejected(self, tmp_path):
        conn = self._conn(tmp_path, "e.db")
        version_no, draft_id = self._publish(conn)
        edition = core.get_edition(conn, version_no)
        conn.execute("DELETE FROM edition_cue WHERE edition_id = ?", (edition["id"],))
        conn.execute("DELETE FROM decision WHERE draft_id = ?", (draft_id,))
        conn.execute("DELETE FROM draft WHERE id = ?", (draft_id,))
        with pytest.raises(core.SourceUnverified, match="源草稿"):
            core.prepare_adjustment(conn, version_no, 5, 9000)
        conn.close()

    def test_archive_tampering_rejected(self, tmp_path):
        conn = self._conn(tmp_path, "f.db")
        version_no, _ = self._publish(conn)
        edition = core.get_edition(conn, version_no)
        conn.execute(
            "UPDATE edition_cue SET end_ms = end_ms + 1 WHERE edition_id = ?",
            (edition["id"],),
        )
        with pytest.raises(core.SourceUnverified):
            core.prepare_adjustment(conn, version_no, 5, 9000)
        conn.close()

    def test_impossible_raises_and_no_silent_output(self, tmp_path):
        conn = self._conn(tmp_path, "g.db")
        version_no, _ = self._publish(conn)
        with pytest.raises(core.AdjustmentImpossible):
            core.prepare_adjustment(conn, version_no, 1, 0)
        conn.close()

    def test_failed_publish_writes_no_archive_rows(self, tmp_path):
        conn = self._conn(tmp_path, "h.db")
        cuts = [core.Cut("A", 1000, 5000, 2)]
        cues = [core.Cue(0, "", 4000, 6000, "被截断")]
        draft_id = core.create_draft(conn, self.VTT, cuts, cues)
        bundle = core.load_bundle(conn, draft_id)
        with pytest.raises(core.PublishBlocked):
            core.publish(conn, draft_id, bundle["row"]["revision"])
        assert conn.execute("SELECT COUNT(*) AS n FROM edition").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM edition_cue").fetchone()["n"] == 0
        conn.close()

    def test_legacy_database_gets_archive_table_on_startup(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        core.init_db(path)
        conn = core.connect(path)
        version_no, _ = self._publish(conn)
        original_blob = core.get_edition(conn, version_no)["vtt_blob"]
        conn.close()
        # 模拟旧版数据库：移除归档表
        conn = core.connect(path)
        conn.execute("DROP TABLE edition_cue")
        conn.close()
        # 应用重启：init_db 只建新表，不动旧数据，原下载内容不变
        core.init_db(path)
        conn = core.connect(path)
        assert core.get_edition(conn, version_no)["vtt_blob"] == original_blob
        assert conn.execute("SELECT COUNT(*) AS n FROM edition_cue").fetchone()["n"] == 0
        # 无归档但源草稿还在 => 走重放路径
        adjusted = core.prepare_adjustment(conn, version_no, 5, 9000)
        assert len(adjusted) == 4
        conn.close()
