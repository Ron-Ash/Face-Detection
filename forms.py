"""
forms.py
────────
Tkinter forms for confirming or creating a tracked person.
All persistence is delegated to weaviate_store and minio_store.
"""

from __future__ import annotations

import tkinter
from tkinter import ttk
from typing import Optional

import numpy as np
import weaviate
from minio import Minio
from PIL import Image, ImageTk

from database import minio_store, weaviate_store
from facialRecognition.trackedFace import TrackedFace



# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize_thumbnail(img: Image.Image, max_size: int = 96) -> Image.Image:
    out = img.copy()
    out.thumbnail((max_size, max_size), Image.LANCZOS)
    return out


def _image_sharpness_score(img: Image.Image) -> float:
    """Variance of the Laplacian — higher means sharper."""
    import numpy as np
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    padded = np.pad(arr, 1, mode="reflect")
    lap = sum(
        kernel[i, j] * padded[i: i + arr.shape[0], j: j + arr.shape[1]]
        for i in range(3) for j in range(3)
    )
    return float(np.var(lap))


def _fetch_display_images(
    wv_client: weaviate.WeaviateClient,
    mn_client: Minio,
    embedding: np.ndarray,
    limit: int = 8,
    distance_threshold: float = 0.45,
) -> list[tuple[Image.Image, float]]:
    """
    1. Ask Weaviate for the closest FaceEmbedding object keys near *embedding*.
    2. Download the corresponding images from MinIO.
    3. Return (image, distance) pairs sorted by sharpness descending.
    """
    key_dist_pairs = weaviate_store.query_embeddings_for_person(
        wv_client, embedding, limit=limit, distance_threshold=distance_threshold
    )
    if not key_dist_pairs:
        return []

    object_keys = [k for k, _ in key_dist_pairs]
    dist_map = {k: d for k, d in key_dist_pairs}

    downloaded = minio_store.download_images_for_person(mn_client, object_keys)
    results = [(img, dist_map[key]) for img, key in downloaded]
    results.sort(key=lambda t: _image_sharpness_score(t[0]), reverse=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# update_form  –  "Is this the right person?"
# ─────────────────────────────────────────────────────────────────────────────

def update_form(
    root: tkinter.Tk,
    wv_client: weaviate.WeaviateClient,
    mn_client: Minio,
    fid: int,
    tracked: TrackedFace,
    pil_img: Optional[Image.Image],
) -> None:
    """
    Show the matched identity alongside the live capture and stored images.

    Layout
    ──────
    Row 0-4  │ col 0 : live capture
             │ col 1-2 : name / affiliation / status / confidence (read-only)
    Row 5    │ col 0-2 : "Stored images" header
    Row 6    │ col 0-2 : scrollable thumbnail strip (from MinIO)
    Row 7    │ col 0-2 : [✓ Yes — add image]  [✗ No — new person]
    """
    window = tkinter.Toplevel(root)
    window.title(f"Confirm Identity — Track {fid}")
    window.resizable(True, True)

    THUMB_SIZE = 80
    CAPTURE_SIZE = 200

    # ── Live capture ──────────────────────────────────────────────────────────
    if pil_img is not None:
        display_img = pil_img.copy()
        display_img.thumbnail((CAPTURE_SIZE, CAPTURE_SIZE), Image.LANCZOS)
        tk_capture = ImageTk.PhotoImage(display_img)
        lbl_capture = tkinter.Label(window, image=tk_capture, relief="solid", bd=1)
        lbl_capture.image = tk_capture
        lbl_capture.grid(row=0, column=0, rowspan=5, padx=10, pady=10, sticky="n")
    else:
        tkinter.Label(window, text="No image", width=20, height=10, relief="solid").grid(
            row=0, column=0, rowspan=5, padx=10, pady=10
        )

    # ── Identity fields ───────────────────────────────────────────────────────
    fp = {"padx": 6, "pady": 3}
    tkinter.Label(window, text=f"Track ID: {fid}", font=("Helvetica", 10, "bold")).grid(
        row=0, column=1, columnspan=2, sticky="w", **fp
    )

    def _ro_field(row: int, label: str, value: str) -> None:
        tkinter.Label(window, text=label, anchor="e", width=12).grid(row=row, column=1, sticky="e", **fp)
        tkinter.Label(window, text=value or "—", anchor="w", fg="#333").grid(row=row, column=2, sticky="w", **fp)

    _ro_field(1, "Name:",        tracked.name or "Unknown")
    _ro_field(2, "Affiliation:", tracked.affiliation or "—")
    _ro_field(3, "Status:",      tracked.status or "—")
    _ro_field(4, "Confidence:",  f"{tracked.confidence:.1f}%" if tracked.confidence is not None else "—")

    # ── Stored images strip ───────────────────────────────────────────────────
    tkinter.Label(
        window, text="Stored images of this person:", font=("Helvetica", 9, "italic")
    ).grid(row=5, column=0, columnspan=3, sticky="w", padx=10, pady=(10, 2))

    strip_frame = tkinter.Frame(window, bd=1, relief="sunken")
    strip_frame.grid(row=6, column=0, columnspan=3, padx=10, pady=4, sticky="ew")
    canvas = tkinter.Canvas(strip_frame, height=THUMB_SIZE + 16, bg="#f0f0f0")
    scrollbar = ttk.Scrollbar(strip_frame, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=scrollbar.set)
    scrollbar.pack(side="bottom", fill="x")
    canvas.pack(side="top", fill="both", expand=True)

    inner = tkinter.Frame(canvas, bg="#f0f0f0")
    canvas.create_window((0, 0), window=inner, anchor="nw")

    def _populate_thumbnails(images: list) -> None:
        """Called on the Tk main thread once the background fetch completes."""
        # Clear the loading indicator
        for w in inner.winfo_children():
            w.destroy()

        if not images:
            tkinter.Label(inner, text="No stored images found.", bg="#f0f0f0", fg="#888").pack(
                side="left", padx=8, pady=8
            )
        else:
            for img, dist in images[:6]:
                thumb = _resize_thumbnail(img, THUMB_SIZE)
                tk_thumb = ImageTk.PhotoImage(thumb)
                col_frame = tkinter.Frame(inner, bg="#f0f0f0")
                col_frame.pack(side="left", padx=4, pady=4)
                lbl = tkinter.Label(col_frame, image=tk_thumb, bg="#f0f0f0")
                lbl.image = tk_thumb  # attach ref to widget — prevents GC
                lbl.pack()
                tkinter.Label(
                    col_frame, text=f"{(1 - dist) * 100:.0f}%",
                    bg="#f0f0f0", fg="#555", font=("Helvetica", 7)
                ).pack()

        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _fetch_in_background() -> None:
        """Runs on a daemon thread — does the blocking network calls, then
        schedules _populate_thumbnails back onto the Tk main thread."""
        try:
            images = _fetch_display_images(wv_client, mn_client, tracked.embedding, limit=8)
        except Exception as e:
            print(f"[update_form] thumbnail fetch error: {e}")
            images = []
        # window.after is thread-safe; it queues the call onto the Tk event loop
        window.after(0, lambda: _populate_thumbnails(images))

    # Show a loading indicator immediately, then kick off the background fetch
    tkinter.Label(inner, text="Loading…", bg="#f0f0f0", fg="#aaa").pack(
        side="left", padx=8, pady=8
    )
    import threading as _threading
    _threading.Thread(target=_fetch_in_background, daemon=True).start()

    # ── Yes / No buttons ──────────────────────────────────────────────────────
    btn_frame = tkinter.Frame(window)
    btn_frame.grid(row=7, column=0, columnspan=3, pady=10)

    def on_yes() -> None:
        """Upload the captured image to MinIO and link it to the person in Weaviate."""
        if pil_img is not None:
            person_uuid = weaviate_store.get_person_uuid_for_embedding(wv_client, tracked.embedding)
            if person_uuid:
                object_key = minio_store.upload_image(mn_client, pil_img, person_uuid)
                weaviate_store.add_face_embedding(wv_client, person_uuid, tracked.embedding, object_key)
                print(f"[update_form] Saved → MinIO:{object_key}, Person:{person_uuid}")
            else:
                print("[update_form] Could not resolve person UUID — image not saved.")
        window.destroy()

    def on_no() -> None:
        window.destroy()
        create_form(root, wv_client, mn_client, fid, tracked, pil_img)

    tkinter.Button(
        btn_frame, text="✓  Yes — add image to this person",
        bg="#2e7d32", fg="white", font=("Helvetica", 10, "bold"),
        padx=10, pady=4, command=on_yes,
    ).pack(side="left", padx=12)

    tkinter.Button(
        btn_frame, text="✗  No — create new person",
        bg="#c62828", fg="white", font=("Helvetica", 10, "bold"),
        padx=10, pady=4, command=on_no,
    ).pack(side="left", padx=12)


# ─────────────────────────────────────────────────────────────────────────────
# create_form  –  Register a brand-new person
# ─────────────────────────────────────────────────────────────────────────────

def create_form(
    root: tkinter.Tk,
    wv_client: weaviate.WeaviateClient,
    mn_client: Minio,
    fid: int,
    tracked: TrackedFace,
    pil_img: Optional[Image.Image],
) -> None:
    """
    Register a new Person.
    On save:
      1. Create Person in Weaviate  → person_uuid
      2. Upload image to MinIO      → object_key
      3. Insert FaceEmbedding in Weaviate linking person_uuid + object_key
    """
    window = tkinter.Toplevel(root)
    window.title(f"New Person — Track {fid}")

    CAPTURE_SIZE = 180

    # ── Live capture ──────────────────────────────────────────────────────────
    if pil_img is not None:
        display_img = pil_img.copy()
        display_img.thumbnail((CAPTURE_SIZE, CAPTURE_SIZE), Image.LANCZOS)
        tk_capture = ImageTk.PhotoImage(display_img)
        lbl = tkinter.Label(window, image=tk_capture, relief="solid", bd=1)
        lbl.image = tk_capture
        lbl.grid(row=0, column=0, rowspan=6, padx=10, pady=10, sticky="n")
    else:
        tkinter.Label(window, text="No image", width=18, height=10, relief="solid").grid(
            row=0, column=0, rowspan=6, padx=10, pady=10
        )

    # ── Editable fields ───────────────────────────────────────────────────────
    fp = {"padx": 6, "pady": 4}

    tkinter.Label(window, text=f"Track ID: {fid}", font=("Helvetica", 10, "bold")).grid(
        row=0, column=1, columnspan=2, sticky="w", **fp
    )

    tkinter.Label(window, text="Name", anchor="e", width=12).grid(row=1, column=1, sticky="e", **fp)
    name_entry = tkinter.Entry(window, width=22)
    name_entry.insert(0, tracked.name or "")
    name_entry.grid(row=1, column=2, sticky="w", **fp)

    tkinter.Label(window, text="Affiliation", anchor="e", width=12).grid(row=2, column=1, sticky="e", **fp)
    aff_entry = tkinter.Entry(window, width=22)
    aff_entry.insert(0, getattr(tracked, "affiliation", "") or "")
    aff_entry.grid(row=2, column=2, sticky="w", **fp)

    tkinter.Label(window, text="Status", anchor="e", width=12).grid(row=3, column=1, sticky="e", **fp)
    status_var = tkinter.StringVar(value=getattr(tracked, "status", "unknown") or "unknown")
    ttk.Combobox(
        window,
        textvariable=status_var,
        values=["approved", "unapproved", "unknown"],
        state="readonly",
        width=19,
    ).grid(row=3, column=2, sticky="w", **fp)

    error_label = tkinter.Label(window, text="", fg="#c62828", font=("Helvetica", 9))
    error_label.grid(row=4, column=1, columnspan=2)

    # ── Save ──────────────────────────────────────────────────────────────────
    def submit() -> None:
        name = name_entry.get().strip()
        if not name:
            error_label.config(text="Name is required.")
            return

        affiliation = aff_entry.get().strip()
        status = status_var.get()

        try:
            # 1. Person → Weaviate
            person_uuid = weaviate_store.create_person(wv_client, name, affiliation, status)

            # 2. Image → MinIO
            img_to_store = pil_img if pil_img is not None else Image.new("RGB", (64, 64))
            object_key = minio_store.upload_image(mn_client, img_to_store, person_uuid)

            # 3. Embedding + key → Weaviate
            weaviate_store.add_face_embedding(wv_client, person_uuid, tracked.embedding, object_key)

            # Update in-memory tracker
            tracked.name = name
            tracked.affiliation = affiliation
            tracked.status = status

            print(f"[create_form] Saved '{name}' → Person:{person_uuid}, MinIO:{object_key}")
            window.destroy()

        except Exception as e:
            error_label.config(text=f"Save failed: {e}")
            print(f"[create_form] Error: {e}")

    tkinter.Button(
        window, text="Save new person",
        bg="#1565c0", fg="white", font=("Helvetica", 10, "bold"),
        padx=10, pady=4, command=submit,
    ).grid(row=5, column=1, columnspan=2, pady=10)