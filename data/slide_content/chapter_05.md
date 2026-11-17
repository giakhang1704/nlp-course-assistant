# Chapter 5

## Auxiliary Verbs and Verb Forms

Auxiliary verbs ràng buộc verb form theo sau: have đi với past participle; be có thể đi với ing-form hoặc past participle trong passive construction. Modal verbs như can và must đi với base form. Trong auxiliary sequence, finite auxiliary đầu tiên mang agreement với subject. Các constraints này giúp grammar chấp nhận những auxiliary sequences hợp lệ.

## Auxiliary Constraints: COMPFORM, MAIN and PASS

Auxiliary verbs ràng buộc dạng của VP theo sau: have đi với past participle, be đi với ing-form hoặc past participle trong passive, còn modal như can và must đi với base form. COMPFORM của auxiliary được chia sẻ với VFORM của VP trong rule VP → (AUX COMPFORM ?s) (VP VFORM ?s). MAIN phân biệt main verb (+) với auxiliary (−); PASS đánh dấu passive construction. Auxiliary còn được nhận diện qua khả năng đứng trước not hoặc trước subject trong yes/no question, như I am not going và Can I try it? Main verb không dùng cùng cấu trúc inversion này mà cần auxiliary phù hợp, thường là do.

## Passive Construction and PASSGAP

Trong passive construction, NP giữ vai trò object của active sentence xuất hiện ở vị trí đầu câu. Agent có thể bị lược bỏ hoặc xuất hiện trong by-phrase. Ví dụ I will hide my hat in the drawer chuyển thành My hat will be hidden in the drawer. Binary feature PASSGAP là + khi constituent thiếu object NP và mặc định là − khi không được chỉ định ở phía trái rule. Hai patterns VP [-PASSGAP] → V [-np] NP và VP [+PASSGAP] → V [-np] phân biệt trường hợp có object tường minh với trường hợp object position còn khuyết.

## Yes/No Questions and Auxiliary Inversion

Subject-auxiliary inversion tạo yes/no question bằng cách đổi vị trí subject NP và auxiliary đầu tiên: Jack is giving Sue a back rub thành Is Jack giving Sue a back rub? Nếu assertion không có auxiliary phù hợp, question dùng một form của do. Đây là local movement. Trong wh-question, constituent được hỏi có thể liên hệ với vị trí khuyết ở xa hơn; trừ trường hợp hỏi chính subject NP, subject và auxiliary cũng đảo vị trí. Who ate the pizza? hỏi subject nên không có cùng inversion như What book did you put on the desk? Wh-words còn có thể hỏi thời gian, địa điểm, manner hoặc ownership.

## Movement, Filler and Gap

Gap là vị trí còn khuyết trong structure; filler là constituent xuất hiện ở vị trí khác nhưng liên hệ với gap đó. Wh-movement tạo liên hệ với wh-constituent được đưa lên trước. Topicalization đưa constituent ra trước để nhấn mạnh; adverb preposing đưa adverbial lên trước; extraposition đưa một số constituents về cuối câu. Filler–gap analysis giữ liên hệ syntax dù surface order đã thay đổi.

## GAP Propagation and Wh-questions

Feature GAP biểu diễn missing constituent và truyền thông tin đó qua các mức của grammar. Với rule có non-lexical head, GAP được chia sẻ giữa mother và head; các subconstituents khác nhận GAP −. Với lexical head, grammar tạo các biến thể để GAP được truyền vào một non-lexical subconstituent tại mỗi biến thể. Ví dụ (S GAP ?g) → (NP GAP −) (VP GAP ?g) truyền gap xuống VP. Nếu rule đã chỉ định GAP thì thuật toán không thêm lại. Tại vị trí phù hợp, empty rule giải quyết missing constituent. Các wh-phrases phải tương thích với category và features của gap, thay vì chỉ được đặt lên đầu câu một cách tùy ý.

## Parsing with GAPs and Empty Constituents

Grammar có GAP có thể chứa productions với phía phải rỗng, nên parser cần xử lý một constituent không chiếm word nào trong input. Khi constituent đang chờ sau dot có GAP pattern khớp với chính constituent đó, parser có thể tạo empty constituent mang EMPTY + và các variable bindings cần thiết. Empty constituent được dùng để mở rộng arc ban đầu. Cơ chế này kết nối xử lý feature constraints với việc nhận diện vị trí khuyết, thay vì đòi tìm một word tường minh ở vị trí gap.

