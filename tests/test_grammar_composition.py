"""Check recombination across typed constituents, not memorized full questions."""
import unittest
from course_assistant.parsing.api import CourseParser


class CompositionalGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = CourseParser()

    def test_quantity_combinations_share_one_query_family_and_keep_slots(self):
        for course in ['NLP', 'Học phần CO3085']:
            for verb in ['có', 'có tổng cộng', 'gồm']:
                for count in ['mấy', 'bao nhiêu']:
                    for unit, attribute in [('tín chỉ', 'credits'), ('chương', 'chapter_count'), ('giờ học', 'total_course_hours')]:
                        question = f'{course} {verb} {count} {unit}?'
                        with self.subTest(question=question):
                            result = self.parser.parse(question)
                            self.assertTrue(result['is_valid'])
                            self.assertFalse(result['ambiguity'])
                            self.assertEqual('QUERY_COURSE_QUANTITY', result['semantic_frame']['query_label'])
                            self.assertEqual(attribute, result['query']['attribute'])

    def test_shared_question_endings_and_typed_concept_phrases(self):
        for question, kind in [('Thuật toán Viterbi có nghĩa là gì?', 'definition'),
                               ('Mô hình CRF có ý nghĩa gì?', 'definition'),
                               ('Thuật toán Viterbi hoạt động ra sao?', 'process'),
                               ('So sánh mô hình CRF và thuật toán Viterbi', 'comparison')]:
            with self.subTest(question=question):
                result = self.parser.parse(question)
                self.assertTrue(result['is_valid'])
                self.assertFalse(result['ambiguity'])
                self.assertEqual(kind, result['semantic_frame']['question_type'])

    def test_count_words_keep_assignment_and_exam_slots(self):
        for question, attribute in [('Final kéo dài mấy phút?', 'duration_minutes'),
                                    ('Bài tập lớn làm theo nhóm mấy người?', 'team_size'),
                                    ('NLP học trong mấy tuần?', 'duration_weeks')]:
            with self.subTest(question=question):
                result = self.parser.parse(question)
                self.assertTrue(result['is_valid'])
                self.assertEqual(attribute, result['query']['attribute'])

    def test_generic_slots_do_not_cross_entity_types(self):
        for question in ['CFG có mấy tín chỉ?', 'Tuần 3 có mấy chương?',
                         'NLP có bao nhiêu người?', 'Final gồm bao nhiêu tín chỉ?',
                         'Môn toán có mấy chương?', 'Có được dùng sử dụng AI không?']:
            with self.subTest(question=question):
                self.assertFalse(self.parser.parse(question)['is_valid'])
