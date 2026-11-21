# BÀI TẬP LỚN XỬ LÝ NGÔN NGỮ TỰ NHIÊN (CO3085)
## Đề tài: Trợ lý hỗ trợ học tập thông minh – Course Assistant
### Hướng tiếp cận: Mở rộng bằng LLM/RAG (Lựa chọn 2) – LLM chạy local qua Ollama

---

### 1. Thành viên nhóm & Phân chia công việc

| STT | Họ và Tên | MSSV | Nhiệm vụ chính |
| :---: | :--- | :--- | :--- |
| 1 | Võ Thiện Đăng Khoa | 2311650 | Văn phạm, sinh câu, parser (Phần I) |
| 2 | Lê Cao Hoài An | 2310012 | Biểu diễn ngữ nghĩa, intent/entity, truy vấn (Phần II) |
| 3 | Vũ Huy Gia Khang | 2311486 | Knowledge base, truy xuất, sinh câu trả lời bằng LLM (4.2) |
| 4 | Nguyễn Đăng Khoa | 2311614 | Ngữ cảnh hội thoại, đánh giá, tích hợp (Phần III, IV) |

- **Mã nguồn:** https://github.com/giakhang1704/nlp-course-assistant
- **Video demo:** https://drive.google.com/file/d/1AAM5hZEWVKRpPROYo9OgukErkdnkkGZY/view?usp=sharing

#### 1.1. Phần I: Viết văn phạm và Parser
| Mục | Nhiệm vụ | File mã nguồn / Cấu hình | Output |
| :---: | :--- | :--- | :--- |
| 3.1 | **Viết tập luật & văn phạm CFG** cho miền Course Assistant, chia theo tầng và theo intent; từ vựng miền tách riêng | `models/grammar/rules/*.cfg`<br>`models/grammar/lexicon.json`<br>`src/course_assistant/parsing/grammar.py` | `output/grammar.txt` |
| 3.2 | **Giải thuật sinh câu**: liệt kê dẫn xuất, xoay vòng giữa các cấu trúc, bỏ câu trùng, tối đa 10.000 câu | `src/course_assistant/parsing/generator.py` | `output/samples.txt` |
| 3.3 | **Bộ phân tích cú pháp Earley**: xuất cây cú pháp hoặc `()` cho từng câu | `src/course_assistant/parsing/tokenizer.py`<br>`src/course_assistant/parsing/parser.py`<br>`src/course_assistant/parsing/api.py` | `output/parse-results.txt` |

#### 1.2. Phần II, III & IV: Diễn giải ngữ nghĩa, RAG, Hội thoại & Đánh giá
| Mục | Nhiệm vụ | File mã nguồn | Output |
| :---: | :--- | :--- | :--- |
| II.1 | **Biểu diễn ngữ nghĩa**: ánh xạ cây cú pháp sang semantic frame và dạng logic | `src/course_assistant/semantics/interpreter.py` | `output/semantic.txt` |
| II.2 | **Nhận diện intent & entity** từ các nút `QUERY_*`, `ATTR_*`, `E_*` của cây | `src/course_assistant/semantics/entities.py`<br>`configs/intent_schema.json` | `output/intent-entity.txt` |
| II.3 | **Truy vấn Knowledge Base**: frame → query có intent, thuộc tính, entity, nguồn ưu tiên | `src/course_assistant/semantics/query_builder.py`<br>`src/course_assistant/rag/question_analysis.py` | `output/query.txt` |
| 4.2 | **Retrieval**: nạp KB, chia chunk, TF-IDF + cosine | `src/course_assistant/knowledge/`<br>`src/course_assistant/retrieval/` | `output/retrieval.txt` |
| 4.2 | **Sinh câu trả lời bằng LLM**: kiểm tra bằng chứng, prompt có citation, kiểm tra đầu ra | `src/course_assistant/rag/policy.py`<br>`src/course_assistant/rag/qa.py`<br>`src/course_assistant/generation/` | `output/answer.txt` |
| III | **Ngữ cảnh hội thoại cục bộ**: lưu trạng thái phiên, giải tham chiếu | `src/course_assistant/dialogue/` | `output/dialogue.txt` |
| IV | **Đánh giá hệ thống**: bộ kiểm thử có nhãn, chấm tự động, phân tích lỗi | `src/course_assistant/evaluation/final.py`<br>`scripts/build_eval_set.py`<br>`data/evaluation/final/` | `output/evaluation.txt` |

---

### 2. Sơ đồ luồng xử lý toàn hệ thống (End-to-End Pipeline)

```
          ┌────────────────────────────┐
          │  Câu hỏi / yêu cầu (input) │
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │         Tokenizer          │ ───► Chuẩn hoá Unicode, dấu câu, tách token
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │    Earley Parser + CFG     │ ───► Cây cú pháp hoặc ()          (Mục 3.1, 3.3)
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │   Semantic Interpreter     │ ───► Semantic frame, dạng logic   (Phần II.1)
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │      Intent / Entity       │ ───► Intent, thuộc tính, thực thể (Phần II.2)
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │       Query Builder        │ ───► Truy vấn theo intent         (Phần II.3)
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │  TF-IDF Retrieval trên KB  │ ───► Top-5 đoạn tài liệu          (4.2)
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │    Kiểm tra bằng chứng     │ ───► Thiếu bằng chứng: từ chối, không gọi LLM
          └─────────────┬──────────────┘
                        ▼
          ┌────────────────────────────┐
          │  LLM qwen3:4b (Ollama)     │ ───► Câu trả lời có nguồn [C1]... (4.2)
          └─────────────┬──────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌────────────────────┐      ┌────────────────────┐
│ Ngữ cảnh hội thoại │      │      Đánh giá      │
│    (Phần III)      │      │     (Phần IV)      │
└────────────────────┘      └────────────────────┘
```

---

### 3. Hướng dẫn cài đặt
Hệ thống viết bằng Python (>= 3.10). Phần I chỉ dùng thư viện chuẩn; Phần II–IV cần thêm scikit-learn, httpx và [Ollama](https://ollama.com/download).

**Cài đặt môi trường:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Tải model và tạo Knowledge Base (một lần):**
```bash
ollama pull qwen3:4b
python -B run.py build-kb
```

---

### 4. Hướng dẫn chạy chương trình

#### 4.1. Chạy riêng Phần I
| Mục | Lệnh chạy tại thư mục gốc | File output | Ghi chú |
| :--- | :--- | :--- | :--- |
| 3.1 | `python -B run.py grammar` | `output/grammar.txt` | Xuất toàn bộ production của văn phạm |
| 3.2 | `python -B run.py generate` | `output/samples.txt` | Sinh tối đa 10.000 câu |
| 3.3 | `python -B run.py parse` | `output/parse-results.txt` | Phân tích `input/sentences.txt` |
| Toàn bộ Phần I | `python -B run.py run` | 3 file trên + `semantic.txt`, `intent-entity.txt`, `query.txt` | Sinh câu, parse lại để kiểm tra, xuất biểu diễn ngữ nghĩa |

#### 4.2. Chạy riêng Phần II, III
| Mục | Lệnh chạy tại thư mục gốc | File output | Ghi chú |
| :--- | :--- | :--- | :--- |
| Xem một câu | `python -B run.py inspect "Tuần 5 học những gì?"` | (in ra màn hình) | Cây cú pháp, intent, entity, dạng logic, query |
| 4.2 | `python -B run.py ask "Môn NLP có bao nhiêu tín chỉ?" --output-dir output/ask` | `answer.txt` | Hỏi một câu |
| 4.2 | `python -B run.py qa-batch --input input/questions.txt --output-dir output/qa` | `retrieval.txt`, `answer.txt` | Mỗi dòng một câu hỏi |
| III | `python -B run.py dialogue --input input/dialogues.jsonl --output-dir output/dialogue` | `dialogue.txt` | Các phiên hội thoại nhiều lượt |
| III | `python -B run.py chat` | (terminal) | Hỏi đáp trực tiếp; `/reset` xoá ngữ cảnh, `/exit` thoát |

#### 4.3. Lệnh chạy toàn bộ hệ thống & Đánh giá
Chạy Phần I, rồi chạy bộ đánh giá (Phần II–IV) và tự xuất `retrieval.txt`, `answer.txt`, `dialogue.txt`, `evaluation.txt`:
```bash
python -B run.py run
python -B run.py evaluate
```
Trên CPU, `evaluate` mất khoảng 40–60 phút; nếu bị ngắt, chạy lại đúng lệnh sẽ tiếp tục từ chỗ dừng.

Chạy unit test (không cần Ollama):
```bash
python -B run.py test
```

#### 4.4. Chạy qua Docker (Môi trường chấm bài của giảng viên)
```bash
docker build -t nlp-course-assistant .
docker run --rm -v $(pwd)/input:/nlp/input -v $(pwd)/output:/nlp/output nlp-course-assistant
```

#### 4.5. Giao diện web demo
Cần Ollama đang chạy trên máy:
```bash
pip install -r requirements-web.txt
cd web/frontend && npm ci && npm run build && cd ../..
python -B serve.py
```
Mở http://127.0.0.1:8000. Hoặc chạy bằng Docker: `docker compose up --build`.

---

### 5. Mô tả chi tiết các thành phần hệ thống

#### 5.1. Văn phạm (Grammar - Mục 3.1)
- Văn phạm **Context-Free Grammar (CFG)** theo hướng *semantic grammar*, chia 14 file trong `models/grammar/rules/`, xuất ra `output/grammar.txt` (1.138 production).
- Ký hiệu xuất phát `S -> OPENER UTTERANCE CLOSER`; `UTTERANCE` phân rã thành các nhóm truy vấn:
  - `QUERY_COURSE_INFO`, `QUERY_COURSE_QUANTITY`: thông tin chung (mã môn, tín chỉ, số chương, giờ học, hình thức và thời lượng thi, trọng số điểm).
  - `QUERY_GET_SCHEDULE`: nội dung từng tuần, chương học vào tuần nào.
  - `QUERY_GET_TOPIC`, `QUERY_DEFINITION`, `QUERY_PROCESS`, `QUERY_COMPARISON`, `QUERY_TOPIC_*`: chủ đề từng chương, khái niệm thuộc chương nào, định nghĩa, so sánh, công thức.
  - `QUERY_GET_ASSIGNMENT`, `QUERY_CHAPTER_EXERCISE`, `QUERY_GET_DEADLINE`: yêu cầu bài tập lớn, bài tập chương, các phần, file cần xuất, thời hạn.
  - `QUERY_GET_RESOURCE`: tài liệu tham khảo.
  - `QUERY_GET_RULE`: quy định học phần, sử dụng AI/LLM.
  - `QUERY_GET_INSTRUCTOR`: giảng viên.
  - `QUERY_GET_LO`: chuẩn đầu ra.
  - `QUERY_CONTEXT`: câu có tham chiếu / tỉnh lược ("Còn chương sau?").
- Mỗi nhóm có nhánh **câu hỏi wh**, **câu hỏi có/không** và **câu yêu cầu**, nên nhận cả "Tuần 5 học gì?", "Bài tập lớn có phần parser không?" lẫn "Liệt kê yêu cầu của phần hỏi đáp."
- Từ vựng miền (môn học, tuần, chương, 104 khái niệm, phần bài tập, quy định, chuẩn đầu ra) nằm trong `lexicon.json`, mỗi từ có kiểu và mã định danh.

#### 5.2. Sinh câu (Mục 3.2)
- Liệt kê dẫn xuất từ `S`, xoay vòng giữa các cấu trúc để câu sinh ra phủ đều mọi loại truy vấn.
- Bỏ câu trùng và câu lặp lại cùng một thực thể; sinh đúng 10.000 câu, mỗi câu được parse lại để kiểm tra.

#### 5.3. Parser cú pháp (Mục 3.3)
- Cài đặt thuật toán **Earley** (top-down chart parsing).
- Câu hợp lệ: xuất cây cú pháp dạng ngoặc trên một dòng. Câu không thuộc văn phạm: xuất `()`.
- `input/sentences.txt`: 106/127 câu parse được (các câu còn lại là câu ngoài phạm vi hoặc sai cấu trúc).

#### 5.4. Semantic Representation (Phần II.1)
- Ánh xạ cây cú pháp sang semantic frame và dạng logic:
  - `Môn NLP có bao nhiêu tín chỉ?` → `GET_COURSE_INFO(attribute=credits, course=CO3085)`
  - `Tuần 5 học những nội dung gì?` → `GET_SCHEDULE(week=WEEK_05)`
  - `Deadline bài tập lớn là khi nào?` → `GET_DEADLINE(attribute=deadline, assignment=BTL01)`

#### 5.5. Intent & Entity (Phần II.2)
- Intent lấy từ nút `QUERY_*`, loại câu từ nhánh ngay dưới nó (wh / có-không / yêu cầu / tỉnh lược), thuộc tính từ `ATTR_*`, thực thể từ các nút từ vựng `E_*` (`configs/intent_schema.json`).
- Các intent: `GET_COURSE_INFO`, `GET_SCHEDULE`, `GET_TOPIC`, `GET_ASSIGNMENT`, `GET_DEADLINE`, `GET_RESOURCE`, `GET_RULE`, `GET_LO`, `GET_INSTRUCTOR`, `CONTEXT_DEPENDENT`.
- Câu có tham chiếu chưa giải ("chương này", "nó") được gắn `CONTEXT_DEPENDENT`.

#### 5.6. Knowledge Base & Truy vấn (Phần II.3)
- KB gồm dữ liệu mẫu (thông tin môn học, lịch 15 tuần, 12 chương, bài tập lớn, quy định, tài liệu), FAQ và 114 mục kiến thức tóm tắt từ slide chương 1–11: tổng **209 tài liệu**.
- Câu parse được tạo truy vấn `semantic` mang intent, thuộc tính, thực thể và nguồn ưu tiên; câu ngoài văn phạm vẫn được tìm bằng câu gốc (`raw_question`).
- Retrieval: **TF-IDF** (unigram + bigram) + cosine, lấy top-5.

#### 5.7. LLM/RAG (Mục 4.2)
- **Model**: `qwen3:4b` chạy local qua Ollama, temperature 0, seed cố định – không cần API key, kết quả tái lập được.
- **Kiểm tra bằng chứng trước khi gọi LLM**: không đủ bằng chứng → trả lời *"Không tìm thấy thông tin này trong dữ liệu môn học."*; câu thiếu ngữ cảnh → hỏi lại.
- **Prompt/Context**: câu hỏi + các đoạn tài liệu đánh nhãn `[C1]`, `[C2]`...; model trả JSON gồm trạng thái, câu trả lời và citation.
- **Kiểm tra đầu ra**: citation phải có trong context và khớp với câu trả lời; sai định dạng hoặc chỉ lặp lại câu hỏi thì sinh lại một lần.

#### 5.8. Ngữ cảnh hội thoại cục bộ (Phần III)
- Lưu trạng thái riêng cho từng phiên: đối tượng đang được nói tới trong 3 lượt gần nhất.
- Giải tham chiếu: *"nó"*, *"chương này"*, *"phần này"*, *"thuật toán này"*, *"nội dung này"*, *"còn chương sau?"*, *"hai cái đó"*; tham chiếu không rõ thì hỏi lại thay vì đoán.

#### 5.9. Lựa chọn phương pháp & Cách đánh giá hệ thống (Phần IV)
- **Lựa chọn hướng tiếp cận**: Nhóm chọn **Lựa chọn 2: Mở rộng bằng LLM/RAG**. Phần hiểu câu (grammar, parser, semantic, intent/entity, truy vấn) và quyết định có trả lời hay không đều do phần NLP của nhóm thực hiện; LLM chỉ viết câu trả lời từ tài liệu đã truy xuất:
  `Natural Language Input -> Tokenizer -> Earley Parser -> Semantic Representation -> Intent/Entity -> Query -> TF-IDF Retrieval -> LLM -> Grounded Answer`.
- **Bộ kiểm thử** (`data/evaluation/final`): **34 câu hỏi** của dữ liệu mẫu (`sample_queries.txt`), **47 câu** của nhóm (câu hợp lệ, các cách diễn đạt khác nhau, câu ngoài văn phạm, câu KB không có thông tin, câu ngoài phạm vi, câu cần ngữ cảnh, câu sai cấu trúc), **5 hội thoại** của dữ liệu mẫu và **3 hội thoại** của nhóm.
- **Cách chấm**: câu có đáp án đúng khi có ý chính của đáp án và trích đúng tài liệu nguồn; câu không có thông tin / ngoài phạm vi đúng khi hệ thống từ chối; câu thiếu ngữ cảnh đúng khi hệ thống hỏi lại.
- **Kết quả (`output/evaluation.txt`)**:
  - **CFG Parsing Accuracy**: `81/81` (`100.00%`)
  - **Intent Accuracy**: `54/54` (`100.00%`)
  - **Entity Accuracy**: `54/54` (`100.00%`)
  - **Retrieval (tài liệu đúng trong top-5)**: `57/58` (`98.28%`)
  - **End-to-End QA Accuracy**: `76/78` (`97.44%`), riêng câu của dữ liệu mẫu `34/34` (`100.00%`)
  - **Hội thoại**: `18/18` lượt (`100.00%`)
  - **Câu ngoài phạm vi & câu KB không có thông tin**: `13/13` từ chối đúng, không sinh thông tin ngoài dữ liệu.

#### 5.10. Các hạn chế của hệ thống
1. **Độ bao phủ của văn phạm**: văn phạm dựa trên luật chỉ phủ các mẫu câu đã viết; câu không dấu, sai chính tả, số viết bằng chữ ("tuần thứ năm") không parse được và phải dựa vào truy xuất bằng câu gốc.
2. **Retrieval dựa trên từ vựng**: TF-IDF không hiểu từ đồng nghĩa, nên dễ trượt khi câu hỏi dùng từ khác hẳn tài liệu.
3. **LLM nhỏ chạy CPU**: `qwen3:4b` mất trung vị khoảng 27 giây mỗi câu (chậm nhất khoảng 88 giây) và đôi khi vẫn từ chối dù tài liệu có thông tin.
4. **Dữ liệu chương 12**: slide bài giảng chỉ có 11 chương nên chương 12 chỉ có phần tóm tắt từ dữ liệu mẫu.
5. **Giải tham chiếu theo mẫu định sẵn**: xử lý tốt hội thoại 2–3 lượt nhưng chưa xử lý tham chiếu mơ hồ giữa nhiều đối tượng cùng loại.

#### 5.11. Nguồn tham khảo
- Đề bài, dữ liệu mẫu, slide bài giảng và mã nguồn mẫu của môn CO3085.
- J. Allen, *Natural Language Understanding*, 2nd ed., Benjamin/Cummings, 1995.
- J. Earley, *An Efficient Context-Free Parsing Algorithm*, Communications of the ACM, 1970. https://doi.org/10.1145/362007.362035
- P. Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Qwen Team, *Qwen3 Technical Report*, 2025. https://arxiv.org/abs/2505.09388; mô hình `qwen3:4b` qua [Ollama](https://ollama.com/library/qwen3).
- Thư viện: scikit-learn (TF-IDF), FastAPI.
