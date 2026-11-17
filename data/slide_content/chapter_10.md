# Chapter 10

## Grammatical Relations for Semantic Interpretation

Parser có thể bỏ bớt chi tiết của parse tree và giữ lại grammatical relations quan trọng cho semantics. Những relations gồm logical subject (LSUBJ), logical object (LOBJ) và indirect object (IOBJ). Semantic interpreter sử dụng các relations này làm input để xác định semantic roles. Cách tổ chức này giảm sự phụ thuộc của Semantic Interpretation vào chi tiết surface syntax.

## Triples and Semantic Pattern Matching

Mỗi triple gồm discourse variable, relation và value. Các relations như PRED, LSUBJ và LOBJ cung cấp thông tin cho semantic pattern matching. Với Jack bought a ticket, grammatical relations được ánh xạ sang event BUYS1 có Jack giữ role AGENT và ticket giữ role THEME. Các patterns khớp constituent categories và positions để tạo những Logical Form fragments tương ứng.

## Verb Classes and Embedded Propositions

Semantic pattern matching có thể dùng verb classes: ACTION-VERB gồm buy, give, steal; ATTITUDE-VERB gồm believe, think; TO-VERB gồm give, donate, throw. Class của predicate giúp xác định cách grammatical relations ánh xạ sang semantic roles. Trong Jill thinks that Jack stole the book, LSUBJ của THINKS1 là Jill và được ánh xạ EXPERIENCER; LOBJ trỏ tới embedded event STEALS1. Với event stole, Jack là AGENT còn book là THEME. Do đó logical subject không phải lúc nào cũng mang role AGENT, và logical object có thể là một proposition/event thay vì chỉ một object.

## Semantic Grammar

Semantic Grammar dùng categories gắn với application domain, như FLIGHT-NP, CITY-NP và FLIGHT-DEST. Các rules FLIGHT-DEST → "to" CITY-NP và FLIGHT-SOURCE → "from" CITY-NP biểu diễn destination và source của flight; TIME-QUERY → "When does" FLIGHT-NP FLIGHT-VP tổ chức câu hỏi thời gian. Categories theo miền giúp giới hạn những combinations không phù hợp mà generic NP rules có thể sinh ra. Parse tree có thể được dùng trực tiếp như Logical Form và chuyển sang database query, hoặc grammar có thể được augment để tạo một Logical Form riêng.

## Semantic Grammar: Advantages and Limitations

Semantic Grammar hỗ trợ xây parser nhanh cho một limited application domain. Vì categories được thiết kế theo domain, chuyển sang domain mới có thể đòi hỏi xây lại semantic grammar. Mở rộng coverage cũng làm tăng khối lượng thiết kế. Khả năng phát triển nhanh trong narrow domain đi kèm hạn chế về portability và broad-domain coverage.

