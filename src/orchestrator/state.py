"""
Farmer Session & State Management.
Maintains persistent farmer profile, environmental state, and multi-turn conversation history.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FarmerProfile:
    """Farmer personal and agronomic profile."""

    farmer_id: str = "farmer_default"
    name: str = "Raitha Bandhu (ರೈತ ಮಿತ್ರ)"
    district: str = "Mandya (Cauvery Basin)"
    language: str = "en"  # "en" or "kn"
    land_area_acres: float = 2.5
    active_crop: Optional[str] = None
    soil_n: float = 75.0  # Nitrogen (kg/ha)
    soil_p: float = 40.0  # Phosphorus (kg/ha)
    soil_k: float = 42.0  # Potassium (kg/ha)
    soil_ph: float = 6.8  # pH


@dataclass
class ChatTurn:
    """A single turn in the conversational advisory stream."""

    turn_id: str
    timestamp: str
    user_text: str
    image_attached: bool
    assistant_response_en: str
    assistant_response_kn: str
    modules_triggered: List[str]
    diagnostic_details: Dict[str, Any] = field(default_factory=dict)
    feedback_rating: Optional[str] = None  # "up", "down", None
    verified_label: Optional[str] = None


class SessionState:
    """Manages profile state and conversation turns."""

    def __init__(self, profile: Optional[FarmerProfile] = None):
        self.profile = profile or FarmerProfile()
        self.history: List[ChatTurn] = []
        self.last_vision_result: Optional[Dict[str, Any]] = None
        self.last_image_hash: Optional[str] = None

    def update_profile(self, **kwargs):
        """Update farmer profile attributes."""
        for k, v in kwargs.items():
            if hasattr(self.profile, k) and v is not None:
                setattr(self.profile, k, v)

    def set_vision_result(self, result: Dict[str, Any], image_hash: Optional[str] = None):
        """Store the latest vision diagnosis and update active crop context."""
        self.last_vision_result = result
        if image_hash:
            self.last_image_hash = image_hash
        if result and result.get("status") == "confident":
            detected_plant = result.get("plant_type", "")
            if detected_plant and detected_plant.lower() != "unknown":
                self.update_profile(active_crop=detected_plant)

    def get_last_diagnosis(self) -> Optional[Dict[str, Any]]:
        """Retrieve most recent vision diagnosis for conversational follow-ups."""
        return self.last_vision_result

    def get_recent_history(self, max_turns: int = 6) -> List[Dict[str, str]]:
        """Return the most recent conversation turns for context-aware multi-turn synthesis."""
        recent = self.history[-max_turns:] if self.history else []
        turns = []
        for t in recent:
            turns.append({
                "role": "user",
                "content": t.user_text,
            })
            turns.append({
                "role": "assistant",
                "content_en": t.assistant_response_en,
                "content_kn": t.assistant_response_kn,
            })
        return turns

    def add_turn(
        self,
        turn_id: str,
        user_text: str,
        image_attached: bool,
        response_en: str,
        response_kn: str,
        modules_triggered: List[str],
        diagnostic_details: Dict[str, Any],
    ) -> ChatTurn:
        """Record turn in history."""
        turn = ChatTurn(
            turn_id=turn_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_text=user_text,
            image_attached=image_attached,
            assistant_response_en=response_en,
            assistant_response_kn=response_kn,
            modules_triggered=modules_triggered,
            diagnostic_details=diagnostic_details,
        )
        self.history.append(turn)
        return turn

    def get_context_summary(self) -> Dict[str, Any]:
        """Return structured summary for context-aware advisory generation."""
        return {
            "profile": asdict(self.profile),
            "recent_turns_count": len(self.history),
            "last_active_crop": self.profile.active_crop,
            "soil_summary": f"N={self.profile.soil_n}, P={self.profile.soil_p}, K={self.profile.soil_k}, pH={self.profile.soil_ph}",
        }

