"""Regression checks for metric semantics and published cohort boundaries."""

import json
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import avsg_metric as metric
from evaluation.parse_output import parse
from evaluation import vocab
from evaluation.score import ROOT, cohort_names, prediction_graph, read_records, score_file


class MetricTests(unittest.TestCase):
    def test_symmetric_relation_and_directional_action(self):
        graph = metric._g([("v1", "man"), ("v2", "woman")], [("v1", "next_to", "v2")])
        reversed_graph = metric._g([("v1", "man"), ("v2", "woman")], [("v2", "next_to", "v1")])
        self.assertEqual(metric.score_pair(graph, reversed_graph)["edge"]["f1"], 1)
        graph["edges"][0]["edge"] = reversed_graph["edges"][0]["edge"] = "watching"
        self.assertEqual(metric.score_pair(graph, reversed_graph)["edge"]["f1"], 0)

    def test_same_class_swap_and_ignored_attributes(self):
        gold = metric._g([("v1", "man"), ("v2", "man"), ("a1", "male_speech")],
                         [("v1", "speaking", "a1")])
        pred = metric._g([("v1", "man"), ("v2", "man"), ("a1", "male_speech")],
                         [("v2", "speaking", "a1")])
        gold["nodes"][0]["attribute"] = "driver"
        pred["nodes"][0]["attribute"] = "passenger"
        self.assertEqual(metric.score_pair(gold, pred)["all"]["f1"], 1)

    def test_oov_prediction_costs_precision(self):
        gold = metric._g([("v1", "man")], [])
        pred = metric._g([("v1", "man"), ("v2", "unlisted_class")], [])
        self.assertAlmostEqual(metric.score_pair(gold, pred)["node"]["f1"], 2 / 3)

    def test_coreference_split_is_penalized(self):
        gold = metric._g([("v1", "man"), ("v2", "cup"), ("a1", "male_speech")],
                         [("v1", "holding", "v2"), ("v1", "speaking", "a1")])
        pred = metric._g([("v1", "man"), ("v2", "man"), ("v3", "cup"), ("a1", "male_speech")],
                         [("v1", "holding", "v3"), ("v2", "speaking", "a1")])
        self.assertEqual(metric.score_pair(gold, pred)["edge"]["f1"], 0.5)

    def test_parser_recovers_both_modality_objects_and_keeps_oov(self):
        text = ('{"nodes":[{"id":"v1","class":"man"}],"edges":[]}\n'
                '{"nodes":[{"id":"a1","class":"unlisted_sound"}],'
                '"edges":[{"source":"v1","edge":"speaking","target":"a1"}]}')
        parsed = parse(text)
        self.assertTrue(parsed["valid"])
        self.assertEqual([node["id"] for node in parsed["nodes"]], ["v1", "a1"])
        self.assertEqual(len(parsed["edges"]), 1)
        self.assertEqual(parsed["oov_nodes"], 1)

    def test_explicit_invalid_raw_record_stays_empty(self):
        graph, valid = prediction_graph({"valid": False, "raw":
            '{"nodes":[{"id":"v1","class":"man"}],"edges":[]}'})
        self.assertFalse(valid)
        self.assertEqual(graph, {"nodes": [], "edges": []})
        with self.assertRaisesRegex(ValueError, "boolean"):
            prediction_graph({"valid": "false", "nodes": [], "edges": []})


class CohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gold = read_records(ROOT / "data/gold/1k_result.jsonl")
        cls.split = json.loads((ROOT / "data/splits/split_v1.json").read_text(encoding="utf-8"))

    def test_source_disjoint_split_and_coverage(self):
        train, test = set(self.split["train_clips"]), set(self.split["test_clips"])
        self.assertEqual((len(train), len(test)), (827, 197))
        self.assertEqual(train | test, set(self.gold))
        self.assertFalse(train & test)
        source = lambda names: {name.rsplit("_", 2)[0] for name in names}
        self.assertFalse(source(train) & source(test))
        self.assertEqual((len(source(train)), len(source(test))), (364, 91))

    def test_canonical_gold_shape_vocabulary_and_endpoints(self):
        self.assertEqual(len(self.gold), 1024)
        for record in self.gold.values():
            self.assertIsInstance(record["caption"], str)
            nodes = record["ground_truth_nodes"]
            self.assertTrue(2 <= len(nodes) <= 10)
            ids = {node["id"] for node in nodes}
            self.assertEqual(len(ids), len(nodes))
            for node in nodes:
                self.assertIsInstance(node["attribute"], str)
                self.assertRegex(node["id"], r"^[va][0-9]+$")
                classes = vocab.VISUAL_CLASSES if node["id"].startswith("v") else vocab.AUDIO_CLASSES
                self.assertIn(node["class"], classes)
            for edge in record["ground_truth_edges"]:
                self.assertIn(edge["source"], ids)
                self.assertIn(edge["target"], ids)
                self.assertIn(edge["edge"], vocab.ALL_EDGES)

    def test_partial_predictions_cannot_silently_reduce_cohort(self):
        path = ROOT / "results/predictions/finetuned/qwen2.5-omni-7b-ft_raw.jsonl"
        with self.assertRaisesRegex(ValueError, "missing 827"):
            score_file(path, self.gold, cohort_names(self.gold))
        result = score_file(path, self.gold, cohort_names(self.gold, "test"), split="test")
        self.assertEqual(result["n_clips"], 197)
        self.assertEqual(f"{result['metrics']['all']['f1']:.3f}", "0.638")

    def test_duplicate_filenames_are_rejected(self):
        with patch.object(Path, "open", return_value=io.StringIO('{"filename":"x.mp4"}\n' * 2)):
            with self.assertRaisesRegex(ValueError, "duplicate filename"):
                read_records("duplicate.jsonl")

    def test_audio_diagnostic_preserves_explicit_a_prefix_rule(self):
        gold = {"x.mp4": {"nodes": [{"id": "a1", "class": "male_speech"}], "edges": []}}
        pred = {"x.mp4": {"nodes": [{"id": "x1", "class": "male_speech"}], "edges": []}}
        with patch("evaluation.score.read_records", return_value=pred):
            result = score_file("unused.jsonl", gold, {"x.mp4"})
        self.assertEqual(result["metrics"]["node"]["f1"], 1)
        self.assertEqual(result["audio_nodes"]["recall"], 0)


if __name__ == "__main__":
    unittest.main()
