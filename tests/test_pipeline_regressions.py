"""Regression checks for evidence admission and bounded discourse state.

Generator output is a fixture; these tests do not measure model answer accuracy.
"""
import json
import unittest
from course_assistant.rag.qa import CourseAssistant
from course_assistant.dialogue.manager import DialogueAssistant


class CitationFixture:
    model = "regression-fixture"
    title = None
    content = "Fixture"

    def generate(self, system, question, context):
        sources = [json.loads(s) for s in context.splitlines()]
        source = next((s for s in sources if s['title'] == self.title), sources[0])
        label = source['citation']
        return {'status': 'answered', 'answer': self.content + f' [{label}]', 'citations': [label]}


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = CitationFixture()
        cls.qa = CourseAssistant(client=cls.client)

    def setUp(self):
        self.client.title, self.client.content = None, 'Fixture'
        self.manager = DialogueAssistant(self.qa)

    def test_evidence_keeps_named_concepts_despite_question_filler(self):
        cases = [
            ('Cho mình hỏi very trong very angry giữ vai trò gì?', 'slide_ch02_adjective_phrases'),
            ('Giải thích cách Viterbi tìm chuỗi tốt nhất.', 'slide_ch06_viterbi_forward'),
            ('Trong chuỗi trợ động từ, have nhận dạng nào?', 'slide_ch05_auxiliaries'),
            ('Term biểu diễn điều gì trong Logical Form?', 'slide_ch07_semantic_units'),
        ]
        for question, document in cases:
            with self.subTest(question=question):
                result = self.qa.ask(question, retrieve_only=True)
                self.assertIn(document, {s['document_id'] for s in result['source_map'].values()})

    def test_assignment_scope_includes_semantics_not_only_parser(self):
        result = self.qa.ask('BTL01 yêu cầu xây dựng những gì?', retrieve_only=True)
        docs = {s['document_id'] for s in result['source_map'].values()}
        self.assertTrue({'assignments.overview', 'assignments.part_1', 'assignments.part_2'} <= docs)

    def test_assignment_part_retains_parent_identity_in_model_context(self):
        result = self.qa.ask('BTL01 có phần hỏi đáp không?', retrieve_only=True)
        blocks = [json.loads(s) for s in result['context'].splitlines()]
        part = next(b for b in blocks if b.get('part') == 2)
        self.assertEqual('BTL01', part['assignment_id'])

    def test_first_checkpoint_is_selected_from_numbered_schedule_milestones(self):
        result = self.manager.ask('Lịch mẫu đặt checkpoint đầu tiên của BTL ở tuần mấy?', retrieve_only=True)
        docs = {s['document_id'] for s in result['source_map'].values()}
        self.assertEqual({'schedule.week_08'}, docs)

    def test_week_inferred_from_actually_cited_schedule(self):
        self.client.title = 'Tuần 12 / Week 12'
        first = self.manager.ask('Lịch mẫu có checkpoint semantic interpretation và QA ở tuần nào?')
        self.assertIn('WEEK_12', {e['id'] for e in first['state_after']['focus']})
        follow = self.manager.ask('Tuần ấy tương ứng chương mấy?', retrieve_only=True)
        self.assertEqual('retrieved', follow['status'])
        self.assertIn('Tuần 12', follow['effective_question'])

    def test_cited_book_is_typed_resource_for_followup(self):
        self.client.title = 'An Introduction to Machine Translation'
        self.manager.ask('Tài liệu dịch máy của Hutchins có tựa gì?')
        follow = self.manager.ask('Cuốn đó ghi năm nào?', retrieve_only=True)
        self.assertEqual('retrieved', follow['status'])
        self.assertIn('An Introduction to Machine Translation', follow['effective_question'])

    def test_local_reflexive_is_not_a_missing_antecedent(self):
        result = self.manager.ask('Một sense trong path-based similarity so với chính nó được điểm bao nhiêu?', retrieve_only=True)
        self.assertEqual('retrieved', result['status'])

    def test_distinct_concepts_sharing_one_document_form_a_pair(self):
        self.client.title = 'Auxiliary Verbs and Verb Forms'
        self.manager.ask('Have đi với dạng nào?')
        self.manager.ask('Còn be đi với dạng nào?')
        result = self.manager.ask('Hai trợ động từ đó có điểm chung gì?', retrieve_only=True)
        names = {e['label'].lower() for e in result['context_used']}
        self.assertEqual({'have', 'be'}, names)

    def test_process_adjacency_comes_from_explicit_source_order(self):
        self.client.title = 'Extractive and Abstractive Summarization'
        self.client.content = 'information ordering'
        self.manager.ask('Bước nào xác định thứ tự các ý trong quy trình tóm tắt?')
        previous = self.manager.ask('Ngay trước bước sắp thứ tự ấy là bước nào?', retrieve_only=True)
        self.assertIn('content selection', previous['effective_question'])
        after = self.manager.ask('Ngay sau bước sắp thứ tự ấy là bước nào?', retrieve_only=True)
        self.assertIn('sentence realization', after['effective_question'])

    def test_missing_references_do_not_fall_through_to_random_retrieval(self):
        for question in ['Nó dùng feature nào?', 'Cuốn đó của ai?', 'Cụm được gạch chân thuộc category nào?']:
            with self.subTest(question=question):
                result = self.qa.ask(question, retrieve_only=True)
                self.assertEqual('clarification_needed', result['status'])
                self.assertEqual([], result['retrieved_chunks'])

    def test_bound_subject_is_not_replaced_by_incidental_vocabulary(self):
        self.client.title = 'Named Entity Recognition (NER)'
        self.manager.ask('NER là gì?')
        self.manager.ask('Trong tác vụ này, token ngoài entity có nhãn gì?')
        follow = self.manager.ask('Ví dụ của tác vụ đó gồm những gì?', retrieve_only=True)
        self.assertEqual('retrieved', follow['status'])
        self.assertEqual({'NER'}, {e['id'] for e in follow['context_used']})

    def test_citation_mismatch_regenerates_once_without_modifying_rejected_answer(self):
        calls = []
        original = self.client.generate
        def generate(system, question, context):
            calls.append(system)
            if len(calls) == 1:
                return {'status': 'answered', 'answer': 'Uncited fixture', 'citations': ['C1']}
            return {'status': 'answered', 'answer': 'Valid fixture [C1]', 'citations': ['C1']}
        self.client.generate = generate
        try:
            result = self.qa.ask('Môn NLP có bao nhiêu tín chỉ?')
        finally:
            self.client.generate = original
        self.assertEqual(2, len(calls))
        self.assertEqual('Valid fixture [C1]', result['answer'])
        self.assertEqual(1, result['generation_validation_retries'])


    def test_enumeration_labels_outside_context_regenerate_once(self):
        calls = []
        original = self.client.generate
        def generate(system, question, context):
            calls.append(system)
            if len(calls) == 1:
                return {'status': 'answered', 'answer': '[C1] một, [C9] hai', 'citations': ['C1']}
            return {'status': 'answered', 'answer': 'Valid fixture [C1]', 'citations': ['C1']}
        self.client.generate = generate
        try:
            result = self.qa.ask('Môn NLP có bao nhiêu tín chỉ?')
        finally:
            self.client.generate = original
        self.assertEqual(2, len(calls))
        self.assertIn('không dùng [Cn] làm số thứ tự', calls[1])
        self.assertEqual('Valid fixture [C1]', result['answer'])


if __name__ == '__main__':
    unittest.main()
