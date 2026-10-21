"""El CSS de la agenda: el de la maqueta del mostrador y la agenda, literal (101b)."""

# ruff: noqa: E501  (CSS literal de la maqueta: las líneas son las de la maqueta, 101b)

AGENDA_CSS = """/* La 101: la agenda */
.agenda-wrap { width: min(1180px, 100%); margin: 0 auto; }
.agenda-head .btn { white-space: nowrap; }
.agenda { display: grid; grid-template-columns: 1fr 340px; gap: 1.25rem; align-items: start; }
.agenda-head { display: flex; align-items: center; gap: .75rem; margin-bottom: 1rem; }
.agenda-head h2 { margin: 0; font-size: 1.15rem; }
.agenda-head .grow { flex: 1; }
.day-nav { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.day-nav button { border: 0; background: var(--surface); padding: .4rem .7rem; font: inherit; cursor: pointer; color: var(--ink-2); }
.day-nav button + button { border-left: 1px solid var(--line); }
.day-nav button:hover { color: var(--brand); }
.grid {
  display: grid; grid-template-columns: 64px repeat(2, 1fr); border: 1px solid var(--line);
  border-radius: var(--r); background: var(--surface); overflow: hidden; box-shadow: var(--e1);
}
.grid .col-head {
  padding: .65rem .8rem; font-weight: 650; font-size: .88rem; border-bottom: 1px solid var(--line);
  background: color-mix(in srgb, var(--brand) 5%, var(--surface));
}
.grid .col-head small { display: block; font-weight: 500; color: var(--muted); font-size: .76rem; }
.grid .hour {
  height: 52px; padding: .25rem .5rem; font-size: .75rem; color: var(--muted);
  border-top: 1px solid var(--line); font-variant-numeric: tabular-nums;
}
.grid .cell { height: 52px; border-top: 1px solid var(--line); border-left: 1px solid var(--line); position: relative; }
.grid .cell.free:hover { background: color-mix(in srgb, var(--brand) 6%, transparent); cursor: pointer; }
.grid .cell.closed { background: repeating-linear-gradient(135deg, transparent 0 6px, color-mix(in srgb, var(--line) 60%, transparent) 6px 7px); }
.appt {
  position: absolute; left: 4px; right: 4px; top: 3px; padding: .3rem .5rem; border-radius: 8px;
  font-size: .8rem; line-height: 1.3; cursor: pointer; z-index: 1;
  background: color-mix(in srgb, var(--brand) 13%, var(--surface));
  border-left: 3px solid var(--brand); color: var(--ink);
}
.appt b { display: block; font-weight: 650; }
.appt span { color: var(--ink-2); }
.appt.done { background: color-mix(in srgb, #1a7f4b 10%, var(--surface)); border-left-color: #1a7f4b; }
.appt.noshow { background: var(--warn-soft); border-left-color: var(--warn); }
.appt.cancel { background: var(--surface); border-left-color: var(--line-2); color: var(--muted); text-decoration: line-through; }
.appt.selected { box-shadow: 0 0 0 2px var(--brand); }
.side { position: sticky; top: 1rem; }
.side .panel-card { padding: 1.1rem 1.2rem; }
.side h3 { margin: 0 0 .2rem; font-size: 1rem; }
.side .lead { margin: 0 0 .9rem; font-size: .86rem; }
.kv { display: grid; grid-template-columns: 92px 1fr; gap: .35rem .75rem; font-size: .88rem; margin: .4rem 0 1rem; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; }
.side .input, .side select { width: 100%; }
.side select {
  font: inherit; padding: .55rem .7rem; border: 1px solid var(--line); border-radius: 10px;
  background: var(--surface); color: var(--ink);
}
.side .label { margin-top: .7rem; }
.slots { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .35rem; }
.slot {
  border: 1px solid var(--line); background: var(--surface); border-radius: 999px; padding: .3rem .7rem;
  font: inherit; font-size: .82rem; cursor: pointer; font-variant-numeric: tabular-nums;
}
.slot[aria-pressed="true"] { background: var(--brand); border-color: var(--brand); color: white; }
.side-actions { display: flex; flex-direction: column; gap: .5rem; margin-top: 1rem; }
.side-actions .row { display: flex; gap: .5rem; }
.side-actions .row .btn { flex: 1; justify-content: center; }
.state { display: inline-block; font-size: .75rem; font-weight: 600; padding: .15rem .55rem; border-radius: 999px; }
.state.booked { background: color-mix(in srgb, var(--brand) 13%, var(--surface)); color: var(--brand); }
.state.cancel { background: var(--off-soft); color: var(--off); }
.event-line { font-size: .78rem; color: var(--muted); margin-top: .8rem; border-top: 1px solid var(--line); padding-top: .6rem; }
.btn-danger { color: var(--off); border-color: color-mix(in srgb, var(--off) 35%, var(--line)); }
.btn-danger:hover { color: white; background: var(--off); border-color: var(--off); }
.confirm[hidden] { display: none; }
.confirm { margin-top: .8rem; padding: .7rem .8rem; border-radius: 10px; background: var(--off-soft); font-size: .86rem; }
.week { display: grid; grid-template-columns: repeat(6, 1fr); gap: .5rem; margin-bottom: 1rem; }
.week button { border: 1px solid var(--line); background: var(--surface); border-radius: 12px; padding: .5rem .6rem; font: inherit; text-align: left; cursor: pointer; }
.week button b { display: block; font-size: .82rem; }
.week button span { font-size: .75rem; color: var(--muted); }
.week button[aria-current="date"] { border-color: var(--brand); box-shadow: 0 0 0 2px color-mix(in srgb, var(--brand) 20%, transparent); }
.week .bar-mini { height: 4px; border-radius: 4px; background: var(--line); margin-top: .35rem; overflow: hidden; }
.week .bar-mini i { display: block; height: 100%; background: var(--brand); }
.grid .cell.blocked { background: repeating-linear-gradient(135deg, transparent 0 6px, color-mix(in srgb, var(--warn) 25%, transparent) 6px 7px); }
.agenda-head .search { width: 16rem; }
.blocked-label { display: block; padding: 4px 6px; font-size: .74rem; color: var(--warn); font-weight: 600; }
.side .list { list-style: none; padding: 0; margin: .5rem 0; }
.side .list li { padding: .35rem 0; border-bottom: 1px solid var(--line); font-size: .85rem; cursor: pointer; }
.side .stats { display: flex; gap: 1.25rem; margin: .6rem 0; }
.side .stats b { display: block; font-size: 1.15rem; }
.side .stats span { font-size: .78rem; color: var(--muted); }
.tabs { margin-bottom: 1rem; }
.convo-wrap { display: grid; grid-template-columns: 280px 1fr 300px; gap: 1rem; align-items: start; }
.threads { border: 1px solid var(--line); border-radius: var(--r); background: var(--surface); overflow: auto; max-height: 70vh; }
.thread { display: block; width: 100%; text-align: left; border: 0; border-bottom: 1px solid var(--line); background: transparent; font: inherit; padding: .65rem .85rem; cursor: pointer; }
.thread b { font-size: .9rem; } .thread small { display: block; color: var(--muted); font-size: .8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.thread[aria-current="true"] { background: color-mix(in srgb, var(--brand) 8%, var(--surface)); }
.thread .count { float: right; }
.convo { display: flex; flex-direction: column; min-width: 0; border: 1px solid var(--line); border-radius: var(--r); background: var(--surface); min-height: 60vh; }
.convo-head { display: flex; align-items: center; gap: .75rem; padding: .7rem 1rem; border-bottom: 1px solid var(--line); }
.convo-head .who { flex: 1; } .convo-head .who b { display: block; } .convo-head .who small { color: var(--muted); }
.msgs { flex: 1; overflow: auto; padding: 1rem; display: flex; flex-direction: column; gap: .5rem; background: color-mix(in srgb, var(--line) 25%, var(--surface)); }
.msg { max-width: 70%; padding: .5rem .75rem; border-radius: 12px; font-size: .9rem; line-height: 1.4; white-space: pre-wrap; }
.msg small { display: block; font-size: .7rem; color: var(--muted); margin-top: .2rem; }
.msg.in { background: var(--surface); border: 1px solid var(--line); align-self: flex-start; }
.msg.out { background: #e0e7ff; align-self: flex-end; }
.compose { display: flex; gap: .5rem; padding: .75rem 1rem; border-top: 1px solid var(--line); }
.compose input { flex: 1; }
"""
