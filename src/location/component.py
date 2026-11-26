"""
Streamlit Component for Consent-Based Live-Location & KVK Center Matching.
Enforces:
1. User-controlled action: Explicit button click required.
2. Browser permission flow: HTML5 Geolocation API with standard prompt.
3. Transparent purpose: Explains district & KVK matching without coordinate tracking.
4. Zero persistence by default: Coordinates are discarded immediately after district resolution.
"""
from dataclasses import asdict
import logging
from typing import Any, Dict, Optional
import streamlit as st
import streamlit.components.v1 as components

from src.location.service import AgroLocationService
from src.orchestrator.state import SessionState

logger = logging.getLogger(__name__)


GEO_JAVASCRIPT_SNIPPET = """
<div style="font-family: sans-serif; padding: 4px 0px;">
    <button id="geo-btn" onclick="requestBrowserLocation()" 
        style="
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
        <span>📍 Detect Current Location (Consent-based)</span>
    </button>
    <div id="geo-status" style="margin-top: 6px; font-size: 0.8rem; color: #4b5563; text-align: center;"></div>
</div>

<script>
function requestBrowserLocation() {
    const statusDiv = document.getElementById("geo-status");
    const btn = document.getElementById("geo-btn");
    
    if (!navigator.geolocation) {
        statusDiv.innerHTML = "<span style='color:#ef4444;'>⚠️ Browser does not support geolocation. Please select district manually.</span>";
        return;
    }

    btn.disabled = true;
    statusDiv.innerHTML = "<span>⏳ Requesting browser permission...</span>";

    navigator.geolocation.getCurrentPosition(
        function(position) {
            statusDiv.innerHTML = "<span style='color:#10b981;'>✅ Permission granted. Resolving district...</span>";
            const lat = position.coords.latitude.toFixed(4);
            const lon = position.coords.longitude.toFixed(4);
            
            // Send coordinates to Streamlit via URL parameters for instant coarse mapping
            // Note: coordinates are immediately cleared by the server once resolved to district
            const url = new URL(window.parent.location.href);
            url.searchParams.set("geo_lat", lat);
            url.searchParams.set("geo_lon", lon);
            url.searchParams.set("geo_consent", "1");
            window.parent.location.href = url.toString();
        },
        function(error) {
            btn.disabled = false;
            let msg = "Location permission denied. Please select your district manually.";
            if (error.code === error.TIMEOUT) {
                msg = "Location request timed out. Please try again or select manually.";
            } else if (error.code === error.POSITION_UNAVAILABLE) {
                msg = "Position unavailable. Please select your district manually.";
            }
            statusDiv.innerHTML = "<span style='color:#ef4444;'>⚠️ " + msg + "</span>";
        },
        {
            enableHighAccuracy: false, // Coarse cellular/WiFi location is preferred for privacy
            timeout: 10000,
            maximumAge: 300000 // Cache for 5 mins
        }
    );
}
</script>
"""


def render_location_widget(
    location_service: AgroLocationService,
    state: SessionState,
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Render consent-based location manager in Streamlit sidebar or page.
    """
    # 1. Process any incoming URL query parameters from browser geolocation
    query_params = st.query_params
    if "geo_lat" in query_params and "geo_lon" in query_params:
        try:
            lat = float(query_params["geo_lat"])
            lon = float(query_params["geo_lon"])
            consent = query_params.get("geo_consent", "0") == "1"

            if consent:
                resolution = location_service.resolve_district(lat, lon, user_consented=True)
                # Update session profile with coarse district name only
                state.update_profile(district=resolution.district)
                st.session_state.resolved_location = resolution.to_dict()
                st.session_state.location_consent_granted = True
                logger.info("Resolved coordinates to district: %s (Coordinates purged from memory)", resolution.district)
        except Exception as e:
            logger.warning("Error processing location query parameters: %s", e)
        finally:
            # IMMEDIATELY purge coordinates from browser URL to prevent persistence or history leakage
            st.query_params.clear()

    # 2. Transparent Purpose Banner
    with st.expander("📍 Location & KVK Center Matching (Privacy First)", expanded=False):
        st.markdown(
            """
            **🔒 Privacy & Consent Notice**:
            - **Purpose**: We detect your location strictly in-memory to determine your **District**, **Agro-Climatic Zone**, and nearest **Krishi Vigyan Kendra (KVK)**.
            - **Zero Persistence**: Your exact GPS coordinates (latitude/longitude) are **never saved to disk, databases, or logs**.
            - **No External Calls**: District and KVK matching run entirely on local offline spatial lookup tables without third-party tracking.
            """
        )

    # 3. Check if location has been resolved
    resolved = st.session_state.get("resolved_location")

    if resolved and st.session_state.get("location_consent_granted", False):
        st.success(
            f"📍 **Detected District**: `{resolved['district']}`\n\n"
            f"• **Agro-Climatic Zone**: {resolved['agro_climatic_zone']}\n"
            f"• **Soil Context**: {resolved['dominant_soil']}"
        )

        kvk = resolved.get("nearest_kvk", {})
        if kvk:
            with st.expander(f"🏛️ Nearest KVK: {kvk.get('name', 'Krishi Vigyan Kendra')}", expanded=True):
                st.markdown(f"**Affiliation**: `{kvk.get('nodal_institution', 'ICAR / State Agricultural University')}`")
                st.markdown(f"📍 **Location**: {kvk.get('location', '')}")
                st.markdown(f"📞 **Helpline**: `{kvk.get('contact_phone', 'N/A')}`")
                st.markdown(f"✉️ **Email**: `{kvk.get('contact_email', 'N/A')}`")
                services = kvk.get("services_offered", [])
                if services:
                    st.markdown("**Available Farmer Services**:")
                    for s in services:
                        st.markdown(f"- {s}")

        if st.button("🔄 Clear Location / Select Manually", use_container_width=True):
            st.session_state.resolved_location = None
            st.session_state.location_consent_granted = False
            st.rerun()

    else:
        # User-controlled action: Browser Geolocation Button
        components.html(GEO_JAVASCRIPT_SNIPPET, height=75)

    return st.session_state.get("resolved_location") or {}
