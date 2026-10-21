"""API tests: auth, role scoping, CSRF, session lifecycle.

Run with:  pytest -q
Uses a throwaway SQLite file and a tiny in-memory dataset, so it neither
touches data/ghoststock.db nor needs generate_data.py to have been run.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import auth
import server

H = {"X-Requested-With": "ghoststock"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setenv("GHOSTSTOCK_ADMIN_EMAIL", "admin@test.local")
    monkeypatch.setenv("GHOSTSTOCK_ADMIN_PASSWORD", "AdminPass123")
    auth._FAILED_LOGINS.clear()
    auth.init_db()

    # tiny dataset: 3 stores x 2 skus x 150 days; S001/SKU1 goes silent at the end
    dates = pd.date_range("2026-01-01", periods=150)
    rows = []
    for st in ("S001", "S002", "S003"):
        for sku in ("SKU1", "SKU2"):
            for i, d in enumerate(dates):
                dead = st == "S001" and sku == "SKU1" and i >= 120
                rows.append((d, st, sku, 0 if dead else 2, 10, 0, ""))
    df = pd.DataFrame(rows, columns=["date", "store_id", "sku_id", "qty_sold",
                                     "qty_on_hand", "promo_active", "promo_name"])
    monkeypatch.setitem(server._state, "synthetic", (df, None, None))
    return TestClient(server.app)


def login(c, email="admin@test.local", pw="AdminPass123"):
    return c.post("/api/auth/login", json={"email": email, "password": pw}, headers=H)


def make_manager(c, stores="S001", email="m@test.local", pw="ManagerPass1"):
    login(c)
    r = c.post("/api/admin/users", headers=H,
               json={"email": email, "password": pw, "role": "manager", "store_ids": stores})
    assert r.status_code == 200, r.text
    c.cookies.clear()


def test_requires_login(client):
    assert client.get("/api/meta").status_code == 401
    assert client.get("/api/admin/users").status_code == 401


def test_bad_password_rejected(client):
    assert login(client, pw="nope").status_code == 401


def test_login_and_me(client):
    assert login(client).status_code == 200
    assert client.get("/api/auth/me").json()["role"] == "admin"


def test_csrf_header_required(client):
    r = client.post("/api/auth/login", json={"email": "admin@test.local", "password": "AdminPass123"})
    assert r.status_code == 403


def test_lockout_after_five_failures(client):
    for _ in range(5):
        login(client, pw="wrong")
    assert login(client).status_code == 429  # even the right password is refused


def test_manager_sees_only_assigned_stores(client):
    make_manager(client, "S001,S002")
    login(client, "m@test.local", "ManagerPass1")
    assert client.get("/api/meta").json()["stores"] == ["S001", "S002"]
    assert client.get("/api/alerts", params={"store": "S003", "date": "2026-05-30"}).status_code == 403
    assert client.get("/api/chart", params={"store": "S003", "sku": "SKU1", "date": "2026-05-30"}).status_code == 403
    assert client.get("/api/alerts", params={"store": "S001", "date": "2026-05-30"}).status_code == 200


def test_manager_cannot_use_admin_endpoints(client):
    make_manager(client)
    login(client, "m@test.local", "ManagerPass1")
    assert client.get("/api/admin/users").status_code == 403
    assert client.post("/api/admin/users", headers=H, json={
        "email": "x@test.local", "password": "Whatever123", "role": "admin"}).status_code == 403


def test_detector_flags_silent_sku(client):
    login(client)
    alerts = client.get("/api/alerts", params={"store": "S001", "date": "2026-05-30"}).json()["alerts"]
    assert [a["sku_id"] for a in alerts] == ["SKU1"]


def test_edit_access_applies_immediately(client):
    make_manager(client, "S001")
    login(client, "m@test.local", "ManagerPass1")
    assert client.get("/api/meta").json()["stores"] == ["S001"]
    mgr_cookies = dict(client.cookies)
    client.cookies.clear()
    login(client)
    uid = [u for u in client.get("/api/admin/users").json() if u["email"] == "m@test.local"][0]["id"]
    assert client.patch(f"/api/admin/users/{uid}", headers=H,
                        json={"role": "manager", "store_ids": "S001,S003"}).status_code == 200
    client.cookies.clear()
    client.cookies.update(mgr_cookies)  # the manager's existing session
    assert client.get("/api/meta").json()["stores"] == ["S001", "S003"]


def test_removed_user_session_dies(client):
    make_manager(client)
    login(client, "m@test.local", "ManagerPass1")
    mgr_cookies = dict(client.cookies)
    client.cookies.clear()
    login(client)
    uid = [u for u in client.get("/api/admin/users").json() if u["email"] == "m@test.local"][0]["id"]
    assert client.delete(f"/api/admin/users/{uid}", headers=H).status_code == 200
    client.cookies.clear()
    client.cookies.update(mgr_cookies)
    assert client.get("/api/meta").status_code == 401


def test_admin_reset_signs_user_out(client):
    make_manager(client)
    login(client, "m@test.local", "ManagerPass1")
    mgr_cookies = dict(client.cookies)
    client.cookies.clear()
    login(client)
    uid = [u for u in client.get("/api/admin/users").json() if u["email"] == "m@test.local"][0]["id"]
    client.post(f"/api/admin/users/{uid}/reset_password", headers=H, json={"password": "BrandNewPass1"})
    client.cookies.clear()
    client.cookies.update(mgr_cookies)
    assert client.get("/api/meta").status_code == 401
    assert login(client, "m@test.local", "BrandNewPass1").status_code == 200


def test_change_own_password(client):
    login(client)
    bad = client.post("/api/auth/change_password", headers=H,
                      json={"current_password": "wrong", "new_password": "Another12345"})
    assert bad.status_code == 400
    ok = client.post("/api/auth/change_password", headers=H,
                     json={"current_password": "AdminPass123", "new_password": "Another12345"})
    assert ok.status_code == 200
    assert client.get("/api/auth/me").status_code == 200  # current session kept
    client.cookies.clear()
    assert login(client, pw="AdminPass123").status_code == 401
    assert login(client, pw="Another12345").status_code == 200


def test_last_admin_protected(client):
    login(client)
    uid = client.get("/api/admin/users").json()[0]["id"]
    assert client.delete(f"/api/admin/users/{uid}", headers=H).status_code == 400
    assert client.patch(f"/api/admin/users/{uid}", headers=H,
                        json={"role": "manager", "store_ids": "S001"}).status_code == 400


def test_sessions_survive_restart(client):
    login(client)
    cookies = dict(client.cookies)
    fresh = TestClient(server.app)  # simulates a new process reading the same DB
    fresh.cookies.update(cookies)
    assert fresh.get("/api/auth/me").status_code == 200


def test_duplicate_email_and_short_password(client):
    login(client)
    body = {"email": "admin@test.local", "password": "Whatever123", "role": "manager", "store_ids": "S001"}
    assert client.post("/api/admin/users", headers=H, json=body).status_code == 400
    body.update(email="new@test.local", password="short")
    assert client.post("/api/admin/users", headers=H, json=body).status_code == 400
