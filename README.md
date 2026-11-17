# Course Assistant – CO3085

Bài tập lớn Xử lý Ngôn ngữ Tự nhiên: trợ lý hỏi đáp về môn học, dùng grammar + parser để hiểu câu hỏi và RAG với LLM chạy local để trả lời.

## Thành viên

| Họ tên | MSSV | Phụ trách |
|---|---|---|
| | | |

## Cài đặt

Python 3.10+, [Ollama](https://ollama.com/download).

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python -B run.py build-kb
```

## Chạy

```bash
python -B run.py run         # Phần I: grammar.txt, samples.txt, parse-results.txt, semantic/intent-entity/query.txt
python -B run.py evaluate    # Phần II–IV: retrieval.txt, answer.txt, dialogue.txt, evaluation.txt
python -B run.py test        # unit test
```

Một số lệnh khác:

```bash
python -B run.py inspect "Tuần 5 học những gì?"                 # xem cây, intent, entity, query của một câu
python -B run.py ask "Môn NLP có bao nhiêu tín chỉ?"             # hỏi một câu
python -B run.py qa-batch --input input/questions.txt --output-dir output/qa
python -B run.py dialogue --input input/dialogues.jsonl --output-dir output/dialogue
python -B run.py chat                                            # hỏi đáp nhiều lượt trên terminal
```

`evaluate` mất khoảng 40–60 phút trên CPU, bị ngắt thì chạy lại sẽ tiếp tục từ chỗ dừng.

Docker (Phần I, theo khung mount của giảng viên):

```bash
docker build -t nlp-course-assistant .
docker run --rm -v "$PWD/input:/nlp/input" -v "$PWD/output:/nlp/output" nlp-course-assistant
```

Web demo: `pip install -r requirements-web.txt`, build giao diện (`cd web/frontend && npm ci && npm run build`), rồi `python -B serve.py` và mở http://127.0.0.1:8000. Hoặc `docker compose up --build` (cần Ollama chạy trên máy).

## Kiến trúc

```
Câu hỏi → tokenizer → Earley parser → cây cú pháp → semantic frame (intent, entity, thuộc tính)
        → query → TF-IDF retrieval → kiểm tra bằng chứng → qwen3:4b (Ollama) → câu trả lời có nguồn
```

```
src/course_assistant/
  parsing/     tokenizer, nạp grammar, sinh câu, Earley parser
  semantics/   semantic frame, dạng logic, entity, query
  knowledge/   nạp KB, chia chunk
  retrieval/   TF-IDF + cosine
  generation/  prompt, client Ollama, kiểm tra citation
  rag/         phân tích câu hỏi, chính sách bằng chứng, điều phối hỏi đáp
  dialogue/    ngữ cảnh hội thoại, giải tham chiếu
  evaluation/  chấm điểm Phần IV
  web/         backend FastAPI
models/grammar/   rules/*.cfg, lexicon.json
configs/          intent_schema.json, rag.json, default.json
data/raw/         dữ liệu môn học + nội dung slide
data/evaluation/  bộ câu hỏi có nhãn (final/ là bộ đánh giá chính)
```

## Grammar và parser

- Grammar CFG theo hướng semantic grammar, chia 14 file trong `models/grammar/rules/`: khung câu, cụm danh từ, vị từ, rồi các cấu trúc theo từng loại truy vấn (thông tin môn học, lịch, chủ đề, bài tập, deadline, tài liệu, quy định, chuẩn đầu ra, ngữ cảnh...).
- Mỗi loại truy vấn có nhánh câu hỏi wh, câu hỏi có/không và câu yêu cầu, nên nhận được cả "Tuần 5 học gì?" lẫn "Nội dung của tuần 5."
- Từ vựng của miền (môn, tuần, chương, khái niệm, quy định...) nằm trong `lexicon.json`. Tổng cộng 1.138 production.
- Sinh câu: liệt kê dẫn xuất, xoay vòng giữa các cấu trúc, bỏ câu trùng, tối đa 10.000 câu.
- Parser: Earley. Câu không thuộc grammar xuất `()`.

## Biểu diễn ngữ nghĩa, intent và entity

Intent lấy từ nút `QUERY_*` của cây, thuộc tính từ nút `ATTR_*`, entity từ các nút từ vựng `E_*` (ánh xạ trong `configs/intent_schema.json`). Kết quả là semantic frame và dạng logic, ví dụ:

```
Môn NLP có bao nhiêu tín chỉ?      → GET_COURSE_INFO(attribute=credits, course=CO3085)
Deadline bài tập lớn là khi nào?   → GET_DEADLINE(attribute=deadline, assignment=BTL01)
```

Câu có từ tham chiếu chưa rõ ("chương này", "nó") được gắn `CONTEXT_DEPENDENT`.

## Knowledge base và truy vấn

KB gồm dữ liệu mẫu của giảng viên (thông tin môn, lịch, 12 chương, bài tập, quy định, tài liệu), FAQ và 114 mục kiến thức từ slide chương 1–11, tổng 209 tài liệu. Câu parse được tạo query có intent, thuộc tính, entity và nguồn ưu tiên; câu ngoài grammar vẫn được tìm bằng câu gốc. Retrieval dùng TF-IDF (unigram + bigram) và cosine, lấy top-5.

## LLM/RAG

Nhóm chọn hướng 4.2 (LLM/RAG). Model `qwen3:4b` chạy local qua Ollama, temperature 0.

- Trước khi gọi LLM, hệ thống kiểm tra bằng chứng: không đủ bằng chứng thì từ chối, câu thiếu ngữ cảnh thì hỏi lại, không gọi model.
- Prompt gồm câu hỏi và các đoạn tài liệu đánh nhãn `[C1]`, `[C2]`...; model trả JSON gồm trạng thái, câu trả lời và citation.
- Đầu ra được kiểm tra: citation phải có trong context và khớp với câu trả lời; sai định dạng hoặc chỉ lặp lại câu hỏi thì sinh lại một lần.
- Hội thoại: lưu đối tượng của các lượt gần nhất để hiểu "chương này", "nó", "chương sau", "hai cái đó".

## Đánh giá

`python -B run.py evaluate` chạy bộ `data/evaluation/final`: 34 câu và 5 hội thoại mẫu của giảng viên, 47 câu và 3 hội thoại của nhóm (câu hợp lệ, cách hỏi khác nhau, câu ngoài grammar, câu không có trong KB, câu ngoài phạm vi, câu cần ngữ cảnh). Câu có đáp án được tính đúng khi có ý chính của đáp án và trích đúng tài liệu nguồn; câu không có thông tin hoặc ngoài phạm vi đúng khi hệ thống từ chối.

Kết quả (`output/evaluation.txt`):

| | Kết quả |
|---|---|
| Parser, intent, entity | 100% |
| Hỏi đáp | 76/78 (34/34 câu của giảng viên) |
| Hội thoại | 18/18 lượt |
| Câu ngoài phạm vi / không có thông tin | từ chối đúng 13/13 |

## Hạn chế

- Grammar chỉ phủ các mẫu câu đã viết; câu không dấu, sai chính tả hoặc số viết bằng chữ ("tuần thứ năm") không parse được.
- TF-IDF dựa trên từ nên dễ trượt khi câu hỏi dùng từ khác hẳn tài liệu.
- Model 4B chạy CPU nên chậm (khoảng 15–60 giây mỗi câu) và đôi khi từ chối dù tài liệu có thông tin.
- Chương 12 chỉ có tóm tắt từ dữ liệu mẫu, không có nội dung slide.
- Hội thoại chỉ xử lý các dạng tham chiếu định sẵn.

## Nguồn tham khảo

- Đề bài, dữ liệu mẫu, slide bài giảng và khung mã của giảng viên môn CO3085.
- Mã tham khảo cho Earley parser và bộ sinh câu (đã chuyển vào `src/course_assistant/parsing/` và mở rộng).
- Ý tưởng tách thành phần từ để hỏi: https://github.com/tantaivo2003/VO-TAN-TAI-BTL-XLNNTN
- J. Earley, "An Efficient Context-Free Parsing Algorithm", 1970.
- Model Qwen3-4B (Apache 2.0), Ollama.
