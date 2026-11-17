# Chapter 1

## Natural Language and Computational Linguistics

Natural Language được nghiên cứu trong Linguistics, Psycholinguistics, Philosophy và Computational Linguistics. Linguistics xem xét cấu trúc ngôn ngữ, cách các từ tạo thành câu và các cách hiểu mà câu cho phép. Psycholinguistics nghiên cứu quá trình con người tạo và hiểu ngôn ngữ, còn Philosophy xem xét ý nghĩa, sự quy chiếu và quan hệ giữa ngôn ngữ với nhận thức. Computational Linguistics xây dựng mô hình ngôn ngữ bằng thuật toán và cấu trúc dữ liệu, kết hợp hiểu biết từ những lĩnh vực này. Môn học hướng đến xây dựng các mô hình Natural Language Understanding có thể triển khai trong những domain cụ thể.

## Natural Language Understanding Applications

Natural Language Understanding có hai nhóm ứng dụng chính: text-based applications và dialogue-based applications. Text-based applications gồm Information Retrieval, Information Extraction, Machine Translation, Summarization và Question Answering. Dialogue-based applications gồm hệ thống hỏi đáp, chăm sóc khách hàng qua điện thoại, tutoring systems, điều khiển máy bằng lời nói và hợp tác giải quyết vấn đề. Dialogue systems cần chủ động tham gia để duy trì cuộc trao đổi tự nhiên và trôi chảy.

## Information Extraction (IE)

Information Extraction tìm và hiểu những phần thông tin liên quan trong văn bản, có thể tập hợp thông tin từ nhiều đoạn. Đầu ra là structured representation, chẳng hạn các relations trong database hoặc một knowledge base. Mục tiêu là tổ chức thông tin hữu ích cho người dùng và biểu diễn thông tin đủ rõ nghĩa để thuật toán tiếp tục inference. Ở mức đơn giản, Information Extraction có thể sử dụng regular expressions và name lists.

## Information Retrieval (IR) and Information Extraction

Information Retrieval tìm các documents phù hợp với một information need từ một document collection lớn. Đầu ra của IR là những documents liên quan để người dùng hoặc bước xử lý tiếp theo khai thác. Information Extraction tập trung lấy thông tin cụ thể từ nội dung và đưa vào structured representation. Vì vậy, IR giải quyết việc tìm documents, còn IE giải quyết việc rút thông tin có cấu trúc từ documents.

## Named Entity Recognition (NER)

Named Entity Recognition là một subtask quan trọng của Information Extraction: tìm và phân loại các tên trong văn bản. Các entity types gồm Person, Date, Location và Organization. Ví dụ, Shen Guofang được gán PER, Reuters được gán ORG, còn những token nằm ngoài entity mang nhãn O. Standard evaluation của NER được thực hiện theo entity, không chỉ theo từng token.

## Sequence Labeling and Maximum Entropy Markov Models

Sequence Labeling gán một label cho mỗi phần tử trong một chuỗi. Chuỗi có thể gồm characters, words, phrases hoặc sentences; các ứng dụng gồm POS Tagging, Named Entity Recognition, Word Segmentation và Text Segmentation. Maximum Entropy Markov Model (MEMM), còn gọi là Conditional Markov Model (CMM), đưa ra từng quyết định dựa trên observations và các quyết định trước đó. Trong POS Tagging, features có thể gồm từ hiện tại, từ trước và sau, một đến ba tags trước đó, suffixes hoặc dấu gạch nối bên trong từ. Search được dùng để khảo sát nhiều chuỗi labels thay vì chỉ một chuỗi quyết định cục bộ.

## Conditional Random Fields (CRF)

Conditional Random Fields (CRF) là whole-sequence conditional model: mô hình xét conditional probability trên cả chuỗi labels. MEMM nối các local classification decisions, còn CRF mô hình hóa chuỗi như một cấu trúc chung. Khi features vẫn mang tính local, conditional sequence likelihood có thể được tính chính xác bằng dynamic programming. CRF tránh causal-competition biases được nêu đối với cách nối local models, nhưng training chậm hơn.

## Greedy Inference for Sequence Labeling

Greedy Inference xử lý chuỗi từ trái sang phải và chọn một label tại mỗi vị trí bằng classifier. Quyết định có thể dùng observations và các labels đã chọn trước đó. Cách này nhanh, dễ triển khai và không cần bộ nhớ bổ sung để giữ nhiều chuỗi ứng viên. Hạn chế là một quyết định sai đã được chốt không thể được sửa lại trong quá trình đi tiếp, nên lỗi trước đó có thể ảnh hưởng các quyết định sau.

## Beam Inference for Sequence Labeling

Beam Inference giữ k chuỗi ứng viên tốt nhất ở mỗi vị trí của input. Mỗi chuỗi được mở rộng bằng các lựa chọn label cục bộ; các kết quả mở rộng cạnh tranh để chiếm k vị trí của bước tiếp theo. Khác với Greedy Inference chỉ giữ một lựa chọn, Beam Inference duy trì nhiều giả thuyết. Cách này dễ triển khai mà không cần dynamic programming, nhưng vẫn là approximate inference: chuỗi tốt nhất toàn cục có thể bị loại khỏi beam ở một bước trung gian.

## Rule-based and Phrase-based Machine Translation

Rule-based Machine Translation xây dictionaries và transformation rules rồi điều chỉnh các rules để cải thiện kết quả dịch. Phrase-based Machine Translation chia input thành các phrases, dịch từng phrase sang ngôn ngữ đích và sắp xếp lại thứ tự các phrases. Ngoài việc lựa chọn bản dịch, hệ thống cần đánh giá fluency của câu đích. Các cách diễn đạt A problem for translation, A problem of translation và A problem in translation minh họa một lựa chọn có thể được xem như bài toán Language Modeling.

## Main Idea and Content Selection in Summarization

Content Selection trong Summarization cần xác định main idea và những supporting details quan trọng. Main idea có thể được nêu trong topic sentence hoặc cần suy ra từ các ý của đoạn. MIDAS tổ chức việc lựa chọn nội dung thành năm thao tác: xác định Main idea; Identify supporting details; Disregard thông tin không quan trọng; Analyze các ý lặp; Simplify, categorize và label các ý quan trọng. Cách lựa chọn này giúp giữ nội dung cốt lõi và giảm redundancy trước khi diễn đạt bản tóm tắt.

## Extractive and Abstractive Summarization

Extractive Summarization tạo bản tóm tắt bằng các phrases hoặc sentences lấy từ source documents. Abstractive Summarization trình bày ý của source documents bằng cách dùng từ khác, ít nhất ở một phần nội dung. Quy trình Summarization gồm ba bước: content selection chọn nội dung cần lấy, information ordering xác định thứ tự trình bày và sentence realization chỉnh lại câu để hình thành bản tóm tắt.

## Question Answering Approaches

Question Answering có các hướng IR-based, knowledge-based và hybrid. IR-based Factoid QA thực hiện question processing để xác định question type, answer type, focus và relations, sau đó tạo search queries. Passage retrieval lấy ranked documents, chia thành passages và rerank; answer processing trích và xếp hạng candidate answers. Knowledge-based QA xây semantic representation của query rồi ánh xạ sang truy vấn trên structured data hoặc resources. Hybrid QA tạo candidate answers bằng IR, bổ sung ontologies hoặc semi-structured data, rồi chấm điểm bằng các knowledge sources phong phú hơn.

## Chatbot Challenges

Chatbot cần hiểu câu hỏi và tạo phản hồi mạch lạc, có ý nghĩa. Việc tạo phản hồi đòi hỏi domain knowledge, discourse knowledge và world knowledge. Chỉ nhận ra các từ trong câu hỏi chưa đủ để bảo đảm một cuộc trao đổi phù hợp. Những loại knowledge này hỗ trợ liên hệ câu hỏi với nội dung và tình huống đang trao đổi.

## Black-box and Glass-box Evaluation

Black-box Evaluation đánh giá performance của toàn hệ thống theo yêu cầu thiết kế mà không xem xét cách các components bên trong hoạt động. Glass-box Evaluation xác định các subcomponents và kiểm tra từng thành phần. Hai cách cung cấp góc nhìn ở mức toàn hệ thống và mức thành phần. Với một hệ thống NLP nhiều bước, kết quả cuối cùng và kết quả của từng bước cần được phân biệt khi evaluation.

## Levels of Language Analysis

Phonetics và Phonology xem xét mối liên hệ giữa từ và âm thanh. Morphology xem xét cách từ được tạo từ các morphemes, như friend kết hợp ly thành friendly. Syntax mô tả cách các từ tạo thành câu đúng và quan hệ cấu trúc giữa các constituents. Semantics nghiên cứu nghĩa của từ và cách chúng kết hợp thành sentence meaning độc lập với context; Pragmatics xem xét cách sử dụng câu trong những tình huống cụ thể ảnh hưởng đến interpretation. Discourse xem xét ảnh hưởng của câu trước lên câu sau, có ích khi hiểu pronouns và thông tin thời gian. World knowledge cung cấp hiểu biết chung về thế giới mà người sử dụng ngôn ngữ cần có.

## Syntax, Logical Form and Knowledge Representation

Để máy tính xử lý meaning của câu và văn bản, hệ thống cần một semantic representation rõ ràng. Một câu có nhiều interpretations có thể cần nhiều representations tương ứng. Các mức biểu diễn gồm syntactic structure, logical form biểu diễn sentence meaning và final meaning representation trong hệ tri thức. Ở mức cuối, syntactic structure và logical form được ánh xạ sang Knowledge Representation (KR). Ngôn ngữ biểu diễn được sử dụng ở mức này là First Order Predicate Calculus (FOPC).

