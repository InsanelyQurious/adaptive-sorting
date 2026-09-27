// Tiny shared mutable channel between the Viewport (mouse drag source) and the TeleopPanel (action composer).
export const teleopBus = { drag: { active: false, dx: 0, dy: 0 }, lastPick: null as null | { u: number; v: number; kind: string; name: string | null; t: number } }
