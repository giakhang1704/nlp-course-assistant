# Chapter 4

## Feature Systems and Augmented Grammar

Feature Systems mô tả syntactic constituents bằng các thuộc tính thay vì chỉ một category name. Augmented Grammar mở rộng CFG bằng các feature structures và các ràng buộc giữa chúng. Ví dụ ART1 (CAT ART ROOT a NUMBER s) mô tả một article có ROOT là a và NUMBER là singular. Feature NUMBER phân biệt s với p; các restrictions có thể biểu diễn agreement giữa article và noun hoặc giữa subject và verb. Nhờ features, grammar mô tả được những khác biệt bên trong cùng một category như NP hoặc V mà không phải tạo một category name riêng cho mọi tổ hợp thuộc tính.

## Feature Variables and Agreement Constraints

Feature variables cho phép nhiều constituents trong một production dùng chung một value. Rule (NP NUMBER ?n) → (ART NUMBER ?n) (N NUMBER ?n) buộc NUMBER của NP, article và noun thống nhất. Khi một constituent xác định value của ?n, các vị trí còn lại phải dùng value tương thích. Trong ví dụ a dog của Chart Parsing, article a xác định AGR là 3s; noun dog cũng có AGR 3s nên arc hoàn tất thành NP AGR 3s. Parser cần tìm variable bindings thỏa feature constraints, không chỉ kiểm tra category names.

## Agreement Feature (AGR)

Agreement có thể phụ thuộc vào person và number. AGR kết hợp hai thuộc tính bằng các values 1s, 2s, 3s, 1p, 2p và 3p; chữ số biểu thị person, s/p biểu thị singular/plural. First person gồm speaker; second person chỉ listener; third person chỉ những đối tượng không phải speaker hoặc listener. Một word form có thể tương thích nhiều AGR values: are có thể nhận 2s, 1p, 2p hoặc 3p, còn fish có thể nhận 3s hoặc 3p. Grammar productions dùng shared variables để chỉ giữ các tổ hợp agreement phù hợp.

## Verb Form (VFORM) and Subcategorization (SUBCAT)

VFORM biểu thị verb form: base như go, pres như goes, past như went, ing như going, pastprt như gone và inf cho infinitive có to. Value fin bao gồm pres hoặc past. SUBCAT mô tả complement pattern mà verb nhận: laugh có -none, take có -np, give có -np-np, want có -vp:inf và tell có -np-vp:inf. Keep có thể nhận -vp:ing, như I keep hoping for the best. VFORM trả lời verb đang ở dạng nào, còn SUBCAT trả lời verb cần hoặc cho phép những complements nào; hai features ràng buộc những khía cạnh khác nhau của Verb Phrase.

## Binary Features and Default Values

Binary feature nhận value + hoặc −. INV biểu thị subject inversion: Jack laughed mang −INV, còn Did Jack laugh mang +INV. Khi tạo constituent, một feature chưa được chỉ định value có thể nhận default value. Cơ chế default giúp tránh khai báo lặp lại mọi feature value trong từng production.

## Lexicon and Head Features

Một word form có thể có nhiều lexical entries. Saw có thể là noun với ROOT SAW1 và AGR 3s, là verb base với ROOT SAW2 và SUBCAT -np, hoặc là past form của SEE1 với SUBCAT -np. Category và ROOT giúp phân biệt các interpretations đó. Head features là những values mà mother constituent chia sẻ với head, chẳng hạn VFORM và AGR trong VP. Khai báo chúng riêng cho phép grammar rút gọn các productions thay vì nhắc lại cùng ràng buộc ở mọi rule. Các giá trị đặc trưng có thể dùng ký hiệu rút gọn như VP [inf].

## Parsing with Features

Trong Chart Parsing với augmented grammar, category names phù hợp chưa đủ để kết hợp hai constituents. Parser cần tìm variable bindings làm các features của expected constituent tương thích với constituent tìm được. Sau đó, parser sao chép arc, áp dụng instantiations và cập nhật chart. Với a dog, NP được hoàn thành khi AGR của article và noun khớp nhau.

