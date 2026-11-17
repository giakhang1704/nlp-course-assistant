import { test } from 'node:test'
import assert from 'node:assert/strict'
import { compactTree, readTree, treeRules, layoutTree } from '../src/parseTree.ts'

test('preserves quoted terminals and epsilon, derives productions from the witness', () => {
  const tree = readTree('(S (PREFIX) (NP "môn" (E "nlp")) (END "?"))')
  assert.ok(tree)
  assert.deepEqual(treeRules(tree), ['S → PREFIX NP END', 'PREFIX → ε', 'NP → "môn" E', 'E → "nlp"', 'END → "?"'])
  assert.equal(readTree('(S "a (b) \\"quoted\\"")')?.children[0].label, 'a (b) "quoted"')
})
test('rejects no-parse and malformed data instead of inventing a tree', () => {
  for (const source of ['()', '(S (NP "x")', '(S "x"))', '(S unquoted)', '(S "x") garbage']) assert.equal(readTree(source), null)
})
test('layout keeps sibling boxes separate and all nodes in bounds', () => {
  const tree = readTree('(S (QUERY_COURSE_CREDITS (NP "môn" "nlp") (COUNT "bao" "nhiêu")) (END "?"))')
  const drawing = layoutTree(tree)
  for (const node of drawing.nodes) {
    assert.ok(node.x - node.width / 2 >= 0)
    assert.ok(node.x + node.width / 2 <= drawing.width)
    for (const other of drawing.nodes) if (other.id > node.id && other.y === node.y) assert.ok(Math.abs(node.x - other.x) >= (node.width + other.width) / 2)
  }
  assert.equal(drawing.edges.length, drawing.nodes.length - 1)
})

test('compact view merges terminal runs without changing original tree or crossing constituents', () => {
  const tree = readTree('(S (PREFIX) (QUERY "có" (COUNT "bao" "nhiêu") "tín" "chỉ") (END))')
  const before = JSON.stringify(tree)
  const compact = compactTree(tree)
  assert.equal(JSON.stringify(tree), before)
  assert.deepEqual(compact.children[1].children.map(c => c.label), ['có', 'COUNT', 'tín chỉ'])
  assert.equal(compact.children[1].children[1].children[0].label, 'bao nhiêu')
  assert.equal(compact.children[0].terminal, false)
  assert.equal(compact.children[0].children.length, 0)
  assert.deepEqual(compactTree(compact), compact)
  const terminals = n => n.terminal ? [n.label] : n.children.flatMap(terminals)
  assert.equal(terminals(tree).join(' '), terminals(compact).join(' '))
  assert.ok(layoutTree(compact).width < layoutTree(tree).width)
})
