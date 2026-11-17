# Chapter 7

## Logical Form and Contextual Interpretation

Semantic Interpretation ánh xạ sentence sang Logical Form để biểu diễn context-independent meaning. Contextual Interpretation ánh xạ Logical Form sang final Knowledge Representation bằng cách liên hệ với situation và knowledge đang có. Pronouns, tense và quantifiers có thể cần context để được hiểu đầy đủ. Discourse cập nhật situation đang được mô tả qua các utterances.

## Lexical Ambiguity, Structural Ambiguity and Quasi-Logical Form

Một word có thể có nhiều senses, như kid chỉ trẻ em hoặc dê con; run có nhiều cách dùng khác nhau. Syntactic structure cũng có thể tạo nhiều interpretations cho cùng word sequence. Quasi-Logical Form có thể giữ các word-sense alternatives hoặc scope alternatives chưa được giải quyết. Final interpretation có thể cần thông tin bổ sung ngoài surface analysis.

## Terms, Predicates and Propositions

Term biểu diễn object hoặc event; predicate biểu diễn property hoặc relation. Áp dụng predicate cho các arguments phù hợp tạo thành proposition. Proper names có thể được biểu diễn bằng terms, common nouns bằng unary predicates và verbs bằng predicates có các arguments tương ứng. Các logical operators NOT, AND và OR kết hợp propositions thành expressions phức tạp hơn.

## Quantifiers, Scope and Plural Readings

Quantifier có thể được biểu diễn theo dạng (quantifier variable: restriction body). MOST dogs bark dùng DOG làm restriction và BARK làm body: đa số objects thỏa DOG cũng thỏa BARK. Quantifier scope gây ambiguity: Every boy loves a dog có thể hiểu mỗi boy yêu một dog nào đó, hoặc có một dog chung mà mọi boy đều yêu. Với plural NP, Two men bought a stereo có distributive reading khi mỗi người mua một stereo và collective reading khi cả hai cùng mua một stereo. Quasi-Logical Form có thể giữ scope chưa quyết định cho bước interpretation tiếp theo.

## Modal Operators and Substitution

Modal operators tạo những contexts trong đó việc thay một term bằng term cùng denotation không luôn bảo toàn meaning được gán cho người nói. Giả sử JACK1 và JOHN2 là cùng người: HAPPY(JACK1) và HAPPY(JOHN2) có thể thay thế trong phát biểu thông thường. Nhưng từ BELIEVE(SUE1, HAPPY(JACK1)) không nhất thiết suy ra BELIEVE(SUE1, HAPPY(JOHN2)), vì Sue có thể không biết hai tên cùng chỉ một người. Vì vậy belief context cần được phân biệt với logical connectives thông thường.

## Tense Operators in Logical Form

Logical Form có thể dùng PAST, PRES và FUT làm tense operators bao quanh proposition. John sees Fido được biểu diễn (PRES (SEES1 JOHN1 FIDO1)); John saw Fido dùng PAST; John will see Fido dùng FUT. Operator cho biết proposition được đặt trong quan hệ thời gian nào, thay vì chỉ thay hình thức của verb. Cách biểu diễn này là một bước trong Logical Form, trước những temporal representations cụ thể hơn ở Knowledge Representation.

## NAME and PRO in Logical Form

NAME và PRO là hai cấu trúc dùng để biểu diễn proper names và pronouns trong Logical Form. NAME có dạng (NAME variable name), như (NAME j1 "John"). PRO có dạng (PRO variable proposition), dùng một variable cùng thông tin về pronoun để chờ diễn giải phù hợp. John ran có thể gắn term NAME vào predicate RUN1 với tense PAST. Trong Every man liked him, pronoun him được giữ bằng PRO thay vì tự chọn ngay một antecedent từ surface sentence.

## Events and Semantic Roles

Event variable cho phép thêm các relations mô tả cùng một event. Trong John broke the window with the hammer, John là AGENT, window là THEME và hammer là INSTR. TO-LOC chỉ destination về địa điểm, TO-POSS chỉ destination về sở hữu; FROM-LOC và FROM-POSS phân biệt nguồn tương ứng. BENEFICIARY là người được hưởng lợi, khác với người nhận object: trong I gave the book to Jack for Susan, Jack là TO-POSS còn Susan là BENEFICIARY. CO-AGENT biểu thị actors cùng thực hiện hành động; CO-THEME có thể mô tả object thứ hai trong một exchange. Các inner roles gắn với nghĩa verb, như AGENT và TO-LOC trong run to the store.

## Location, Possession and Time Roles

Semantic roles biểu diễn những quan hệ cụ thể của event hoặc state. AT-LOC mô tả vị trí, như box contains a ball; AT-POSS mô tả possession, như John owns a book. PATH-LOC mô tả đường đi, như The bird flew from the lake along the river gorge. AT-TIME chỉ một thời điểm, như at 3 o’clock; FROM-TIME và TO-TIME chỉ điểm đầu/cuối, như I worked from one until three. Các roles còn áp dụng cho state changes: The temperature remained at zero dùng AT-VALUE, còn The temperature rose from zero dùng FROM-VALUE. Tên preposition giống nhau không bảo đảm cùng semantic role: to có thể đánh dấu destination địa điểm hoặc possession.

## Speech Acts

Speech-act operators phân biệt việc khẳng định, hỏi và yêu cầu hành động ngay cả khi nội dung event liên quan tương tự. ASSERT bao proposition được khẳng định; Y/NQUERY bao proposition cần kiểm tra; COMMAND biểu thị action cần thực hiện; WHQUERY chỉ đối tượng cần xác định. The man ate a peach tương ứng ASSERT, Did the man eat a peach? tương ứng Y/NQUERY, còn Eat the peach! dùng COMMAND. Trong wh-question, who có thể biểu diễn bằng term dạng <WH p1 PERSON>, còn which man thêm restriction MAN. Những restrictions này xác định loại answer cần tìm.

