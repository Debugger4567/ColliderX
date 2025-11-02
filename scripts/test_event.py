from physics.collision import init_event_db, simulate_event

init_event_db()
eid = simulate_event("Higgs", n_bodies=2)
print("Event stored with ID:", eid)