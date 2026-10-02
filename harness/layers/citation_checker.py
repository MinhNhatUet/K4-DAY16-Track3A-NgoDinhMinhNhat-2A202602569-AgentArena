"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import json
import re
import unicodedata

from arena.model import parse_output

from harness.middleware import Middleware

#: = arena.scorer.MAX_CLAIM_CHARS; dài hơn là OVERLONG, đừng đòi.
MAX_LINE_CHARS = 500

#: Phần đuôi bị bỏ phải đủ dài mới đáng tốn một lượt gọi model.
MIN_MISSING_CHARS = 20


#: = arena.scorer.MIN_SUPPORT_CHARS.
MIN_QUOTE_CHARS = 12


def _norm(text: str) -> str:
    # Cùng dạng so sánh với arena.scorer._norm: model thật hay viết thường.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text).casefold()).strip()


def in_one_line(text: str, doc) -> bool:
    """Luật SUPPORTED của scorer: `text` (đã chuẩn hoá, 12-500 ký tự) nằm
    gọn trong một dòng của `doc`. Chỉ để so khớp, không ghi vào claim."""
    t = _norm(text)
    if not MIN_QUOTE_CHARS <= len(t) <= MAX_LINE_CHARS:
        return False
    return any(t in _norm(line) for line in doc.body.splitlines())

#: = arena.runner.MAX_SEARCH_K; xin nhiều hơn thì runner cũng kẹp về 10.
SEARCH_K = 10

#: Model thật hay trích nửa dòng, rơi mất key term ở nửa sau (đo trên
#: pub-08 với gemini: claim ĐÚNG nhưng thiếu "tối đa là 10%" -> recall 0).
#: Layer không được nối chữ vào claim, nên chỉ có thể nhắc trước khi FINAL.
#: Không chứa doc_id hay FINALIZE_SENTINEL để MockModel bỏ qua nó.
QUOTE_REMINDER = (
    "Chỉ áp dụng KHI bạn viết FINAL (chưa đủ bằng chứng thì cứ tiếp tục "
    "search/fetch_doc): mỗi claim phải là TOÀN BỘ một dòng chép nguyên văn "
    "từ tài liệu đã đọc, từ đầu đến cuối dòng, không cắt bớt, không sửa "
    "chữ hay dấu câu; doc_id là tài liệu chứa đúng dòng đó."
)


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def wrap_tool_call(self, ctx, call, name, args):
        if name == "search" and isinstance(args, dict):
            # Model thật đặt truy vấn ngắn; tài liệu đúng hay rơi ở hạng 6-8
            # (pub-08: "an toàn lao động" -> doc-0017 hạng 6). Lấy tới trần
            # của runner; scorer không phạt độ rộng, chỉ phạt token.
            args = {**args, "k": SEARCH_K}
        result = call(name, args)
        if name == "fetch_doc" and result.ok:
            ctx.state["citation_fetched"] = True
        return result

    def before_model(self, ctx, messages):
        # Chỉ khi đã có tài liệu trong tay: nhắc sớm hơn làm model thật chốt
        # FINAL ngay sau search (đo được: abstain, S 30 -> 20). Một fetch đã
        # xảy ra cũng bảo đảm có lượt assistant, nên MockModel không nhầm
        # lời nhắc là câu hỏi của brief.
        if ctx.state.get("citation_fetched"):
            return messages + [{"role": "user", "content": QUOTE_REMINDER}]
        return messages

    def wrap_model_call(self, ctx, call, messages):
        # Lời nhắc chung chưa đủ: gemini vẫn dừng ở dấu chấm đầu tiên của một
        # dòng hai câu, rơi key term ở câu sau (pub-08, 2/2 lượt). Scorer gộp
        # mọi FINAL đã ghi trace để xét provenance, nên hỏi lại MỘT lần là
        # hợp lệ: chữ trong claim vẫn do model tự viết.
        response = call(messages)
        if ctx.state.get("citation_reasked") or ctx.corpus is None:
            return response
        lines = self._partial_quotes(ctx, response.text)
        if not lines:
            return response
        ctx.state["citation_reasked"] = True
        # Đưa sẵn object JSON: chỉ nói "chép cả dòng" thì gemini vẫn tách
        # một dòng hai câu thành hai claim (pub-02), mỗi claim thiếu key term.
        feedback = (
            "FINAL của bạn chỉ chép MỘT PHẦN dòng tài liệu, nên thiếu dữ kiện. "
            "Viết lại FINAL, giữ nguyên định dạng. Mỗi dòng dưới đây phải là "
            "MỘT claim DUY NHẤT chứa NGUYÊN CẢ DÒNG — không tách theo câu, "
            "không cắt bớt. Thay các claim chỉ chép một phần dòng bằng đúng "
            "các claim sau:\n"
            + "\n".join(
                json.dumps({"text": line, "doc_id": doc_id}, ensure_ascii=False)
                for doc_id, line in lines
            )
        )
        return call(messages + [
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": feedback},
        ])

    @staticmethod
    def _partial_quotes(ctx, text):
        """(doc_id, dòng đầy đủ) cho mỗi claim chỉ là một khúc của dòng đó."""
        from harness.agent import _canonicalise  # agent không import layers

        parsed = parse_output(_canonicalise(text))
        final = parsed.final if parsed.kind == "final" else None
        claims = final.get("claims") if isinstance(final, dict) else None
        if not isinstance(claims, list):
            return []
        observed = ctx.observed_text
        read = [d for d in ctx.corpus.docs if d.body in observed]
        found = []
        for claim in claims:
            quote = _norm(claim.get("text", "")) if isinstance(claim, dict) else ""
            if not quote or any(
                quote == _norm(l) for d in read for l in d.body.splitlines()
            ):
                continue
            for doc in read:
                line = next((
                    l.strip() for l in doc.body.splitlines()
                    if quote in _norm(l)
                    and len(_norm(l)) - len(quote) >= MIN_MISSING_CHARS
                    and len(l.strip()) <= MAX_LINE_CHARS
                ), None)
                if line is not None:
                    if (doc.doc_id, line) not in found:
                        found.append((doc.doc_id, line))
                    break
        return found

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report
        observed = ctx.observed_text
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue
            doc_id = claim.get("doc_id")
            doc = ctx.corpus.get(doc_id.strip()) if isinstance(doc_id, str) else None
            if doc is not None and in_one_line(text, doc):
                continue
            for doc in ctx.corpus.docs:
                if doc.body in observed and in_one_line(text, doc):
                    claim["doc_id"] = doc.doc_id
                    break
        report["citations"] = sorted({
            c["doc_id"] for c in claims
            if isinstance(c, dict) and isinstance(c.get("doc_id"), str)
        })
        return report
