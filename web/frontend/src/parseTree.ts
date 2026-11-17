export type ParseNode = { label: string; terminal: boolean; children: ParseNode[] }

// The parser emits quoted JSON terminals and parenthesized nonterminals.
// Empty parentheses mean no parse; a childless nonterminal means epsilon.
export function readTree(source: string): ParseNode | null {
  if (!source || source.trim() === '()' || source.length > 100000) return null
  const tokens = source.match(/"(?:\\.|[^"\\])*"|\(|\)|[^\s()"]+/g) || []
  let at = 0, count = 0
  function read(depth: number): ParseNode {
    if (depth > 100 || ++count > 2000 || tokens[at++] !== '(') throw Error('Invalid tree')
    const label = tokens[at++]
    if (!label || ['(', ')'].includes(label) || label.startsWith('"')) throw Error('Invalid label')
    const children: ParseNode[] = []
    while (at < tokens.length && tokens[at] !== ')') {
      if (tokens[at] === '(') children.push(read(depth + 1))
      else {
        const token = tokens[at++]
        if (!token.startsWith('"')) throw Error('Expected quoted terminal')
        children.push({ label: JSON.parse(token), terminal: true, children: [] })
      }
    }
    if (tokens[at++] !== ')') throw Error('Unclosed tree')
    return { label, terminal: false, children }
  }
  try { const tree = read(0); return at === tokens.length ? tree : null } catch { return null }
}

export function treeRules(tree: ParseNode): string[] {
  const rules = new Set<string>()
  function walk(node: ParseNode) {
    if (node.terminal) return
    rules.add(`${node.label} → ${node.children.map(c => c.terminal ? JSON.stringify(c.label) : c.label).join(' ') || 'ε'}`)
    node.children.forEach(walk)
  }
  walk(tree)
  return [...rules]
}

// Display-only compaction, preserving constituent boundaries and epsilon.
// Terminal runs are not a linguistic Vietnamese word-segmentation result.
export function compactTree(tree: ParseNode): ParseNode {
  const children: ParseNode[] = []
  for (const child of tree.children) {
    const previous = children.at(-1)
    if (child.terminal && previous?.terminal) previous.label += ` ${child.label}`
    else children.push(compactTree(child))
  }
  return { ...tree, children }
}

export function layoutTree(tree: ParseNode) {
  const nodes: { node: ParseNode; x: number; y: number; width: number; id: number }[] = []
  const edges: { x1: number; y1: number; x2: number; y2: number }[] = []
  const widths = new Map<ParseNode, number>()
  const boxWidth = (n: ParseNode) => Math.max(66, (n.label.length + (!n.terminal && !n.children.length ? 4 : 0)) * 8 + 30)
  function measure(n: ParseNode): number {
    const width = Math.max(boxWidth(n), n.children.reduce((sum, c) => sum + measure(c), 0) + Math.max(0, n.children.length - 1) * 18)
    widths.set(n, width); return width
  }
  const width = measure(tree) + 48
  function place(n: ParseNode, left: number, depth: number) {
    const x = left + widths.get(n)! / 2, y = 32 + depth * 92
    nodes.push({ node: n, x, y, width: boxWidth(n), id: nodes.length })
    const childrenWidth = n.children.reduce((sum, c) => sum + widths.get(c)!, 0) + Math.max(0, n.children.length - 1) * 18
    let childLeft = x - childrenWidth / 2
    for (const c of n.children) {
      const cx = childLeft + widths.get(c)! / 2
      edges.push({ x1: x, y1: y + 19, x2: cx, y2: y + 92 - 19 })
      place(c, childLeft, depth + 1); childLeft += widths.get(c)! + 18
    }
  }
  place(tree, 24, 0)
  return { nodes, edges, width, height: Math.max(...nodes.map(n => n.y)) + 38 }
}
