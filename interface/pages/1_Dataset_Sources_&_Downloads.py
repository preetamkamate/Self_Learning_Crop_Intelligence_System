"""
Streamlit Page: Dataset Sources & Downloads.
Displays the vetted agricultural dataset registry for whole-plant and multi-part crop health.
Provides safe, on-demand downloading of approved research datasets, checksum verification,
and responsible access instructions for gated resources.
"""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from download_datasets import (
    download_dataset,
    get_dataset_by_id,
    list_datasets,
    load_registry,
)

st.set_page_config(
    page_title="Dataset Sources & Downloads | Crop Intelligence System",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Approved Agricultural Dataset Registry & Downloads")
st.markdown(
    """
    This portal catalogues legally vetted, open-access, and research-licensed agricultural datasets
    supporting **whole-plant, canopy, stem, fruit, root, bark/trunk, and nutrient deficiency** diagnosis.
    
    *All datasets follow strict open research licenses (CC BY 4.0, CC0 1.0, or Academic Use). 
    No downloads occur automatically at application startup.*
    """
)

# Load current datasets
datasets = list_datasets()

# ------------------------------------------------------------------------------
# 1. Summary Metrics
# ------------------------------------------------------------------------------
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
total_datasets = len(datasets)
ready_count = sum(1 for d in datasets if d.get("status") == "ready")
total_size_mb = sum(d.get("size_mb", 0) for d in datasets)
total_images = sum(d.get("expected_images", 0) for d in datasets)

m_col1.metric("Catalogued Datasets", total_datasets)
m_col2.metric("Ready Locally", ready_count)
m_col3.metric("Total Archive Volume", f"{total_size_mb / 1024:.2f} GB")
m_col4.metric("Annotated Images", f"{total_images:,}")

st.markdown("---")

# ------------------------------------------------------------------------------
# 2. Dataset Registry Overview Table
# ------------------------------------------------------------------------------
st.subheader("📋 Dataset Registry Catalog")

table_rows = []
for d in datasets:
    status_icon = "🟢 Ready" if d.get("status") == "ready" else ("🟡 In Progress" if d.get("status") == "downloading" else "⚪ Not Downloaded")
    table_rows.append({
        "Dataset ID": d["id"],
        "Dataset Name": d["name"],
        "License": d.get("license", "").split("(")[0].strip(),
        "Crop Coverage": ", ".join(d.get("crops", [])),
        "Supported Viewpoints / Parts": ", ".join(d.get("supported_parts", [])),
        "Available Labels": ", ".join(d.get("labels_available", [])),
        "Size": f"{d.get('size_mb', 0)} MB",
        "Images": f"{d.get('expected_images', 0):,}",
        "Status": status_icon,
        "Source Page": d.get("official_page", ""),
    })

df = pd.DataFrame(table_rows)

st.dataframe(
    df[[
        "Dataset ID", "Dataset Name", "Status", "Size", "Images",
        "Supported Viewpoints / Parts", "Crop Coverage", "Available Labels", "License"
    ]],
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ------------------------------------------------------------------------------
# 3. Interactive Dataset Inspector & On-Demand Downloader
# ------------------------------------------------------------------------------
st.subheader("📥 Inspect & Download Approved Datasets")

selected_id = st.selectbox(
    "Select an approved dataset to inspect or download:",
    options=[d["id"] for d in datasets],
    format_func=lambda x: f"{x} - {next((d['name'] for d in datasets if d['id'] == x), '')[:60]}...",
)

selected_ds = get_dataset_by_id(selected_id)

if selected_ds:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown(f"### {selected_ds['name']}")
        st.markdown(f"**Official Source Page**: [{selected_ds['official_page']}]({selected_ds['official_page']})")
        st.markdown(f"**License**: `{selected_ds.get('license')}`")
        st.markdown(f"**Academic Citation**:\n> *{selected_ds.get('citation')}*")
        st.markdown(f"**Known Agronomic Limitations**:\n> {selected_ds.get('known_limitations', 'None recorded.')}")

    with c2:
        st.markdown("### Specifications")
        st.write(f"- **Plant Parts**: `{', '.join(selected_ds.get('supported_parts', []))}`")
        st.write(f"- **Crops**: `{', '.join(selected_ds.get('crops', []))}`")
        st.write(f"- **Labels**: `{', '.join(selected_ds.get('labels_available', []))}`")
        st.write(f"- **Size**: `{selected_ds.get('size_mb')} MB` (~`{selected_ds.get('expected_images'):,}` images)")
        st.write(f"- **Local Directory**: `{selected_ds.get('local_dir')}`")
        st.write(f"- **Current Status**: `{selected_ds.get('status', 'not_downloaded').upper()}`")

    # Download Button Logic
    st.markdown("#### Download Action")
    if selected_ds.get("status") == "ready":
        st.success(f"✅ This dataset is already downloaded and ready at `{selected_ds.get('local_dir')}`.")
        if st.button("Force Re-Download Dataset", key=f"force_{selected_id}"):
            with st.spinner(f"Re-downloading {selected_ds['name']} from official mirror..."):
                res = download_dataset(selected_id, force=True)
                if res["status"] == "ready":
                    st.success(res["message"])
                    st.rerun()
                else:
                    st.error(res["message"])
    elif selected_ds.get("requires_auth"):
        st.warning("🔒 **Manual Authentication / Registration Required**")
        st.info(
            f"This dataset ({selected_ds['name']}) requires acceptance of terms or account login.\n\n"
            f"**Setup Instructions**:\n{selected_ds.get('auth_instructions')}\n\n"
            "This application respects user privacy, logins, and terms of service, and will never attempt to bypass credentials."
        )
    else:
        st.info("💡 Downloads are verified against SHA-256 checksums and saved to isolated directories in `data/raw/datasets/`.")
        if st.button("Download Approved Dataset", key=f"dl_{selected_id}"):
            with st.spinner(f"Safely downloading {selected_ds['name']} ({selected_ds.get('size_mb')} MB)..."):
                res = download_dataset(selected_id)
                if res["status"] == "ready":
                    st.success(res["message"])
                    st.rerun()
                elif res["status"] == "requires_auth":
                    st.warning(res["message"])
                else:
                    st.error(f"Download failed: {res.get('message')}")

st.markdown("---")

# ------------------------------------------------------------------------------
# 4. Legal, Ethics & Primary Sourcing Guidelines
# ------------------------------------------------------------------------------
st.subheader("🏛️ Legal Compliance, Ethics & Extension Citations")
st.markdown(
    """
    1. **Strict License Compliance**: Only datasets under permissive research licenses (CC BY 4.0, CC0 1.0) or authorized academic access are included.
    2. **Primary Agricultural Sourcing**: Management practices, chemical dosages, and bio-control guidelines are cited strictly from:
       - **ICAR** (Indian Council of Agricultural Research, New Delhi)
       - **ICAR-IIHR** (Indian Institute of Horticultural Research, Bengaluru)
       - **ICAR-CPCRI** (Central Plantation Crops Research Institute, Kasaragod)
       - **ICAR-CCRI** (Central Citrus Research Institute, Nagpur)
       - **UAS Bengaluru & UAS Dharwad** (Karnataka State Agricultural Universities)
    3. **Responsible Pesticide Use**: Chemical dosages adhere to **CIBRC** (Central Insecticides Board & Registration Committee) approved labels. No unverified chemical formulations are generated.
    """
)
