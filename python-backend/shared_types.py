# python-backend/shared_types.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import date, datetime

# =========================
# CONTEXT
# =========================
class AirlineAgentContext(BaseModel):
    """
    Context for airline customer service agents,
    now expanded to include conference and potential future domains.
    This context is shared across all agents.
    """
    # --- Existing Airline/Customer Fields ---
    passenger_name: Optional[str] = None
    confirmation_number: Optional[str] = None
    seat_number: Optional[str] = None
    flight_number: Optional[str] = None
    account_number: Optional[Union[str, int]] = None
    customer_id: Optional[str] = None
    booking_id: Optional[str] = None
    flight_id: Optional[str] = None
    customer_email: Optional[str] = None
    customer_bookings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- Conference & Delegate-Specific Fields ---
    is_conference_attendee: Optional[bool] = False
    conference_name: Optional[str] = None
    user_conference_role: Optional[str] = None  # e.g., "attendee", "speaker", "organizer"
    user_job_title: Optional[str] = None
    user_company_name: Optional[str] = None
    user_bio: Optional[str] = None
    user_social_media_links: Optional[Dict[str, str]] = Field(default_factory=dict)
    user_contact_info: Optional[Dict[str, str]] = Field(default_factory=dict)
    user_registered_tracks: List[str] = Field(default_factory=list)
    user_conference_interests: List[str] = Field(default_factory=list)
    user_personal_schedule_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    # --- Additional User Fields (Missing from original) ---
    user_location: Optional[str] = None
    user_registration_id: Optional[Union[str, int]] = None
    user_conference_package: Optional[str] = None
    user_primary_stream: Optional[str] = None
    user_secondary_stream: Optional[str] = None
    
    # --- Placeholder for future Networking/Business Fields ---
    # last_networking_search_results: List[Dict[str, Any]] = Field(default_factory=list)
    # current_viewed_business_details: Optional[Dict[str, Any]] = None

    @validator('account_number', 'user_registration_id', pre=True)
    def convert_to_string(cls, v):
        """Convert numeric values to strings."""
        if v is not None:
            return str(v)
        return v

# =========================
# GUARDRAIL OUTPUT TYPES
# =========================
class RelevanceOutput(BaseModel):
    """Schema for relevance guardrail decisions."""
    reasoning: Optional[str]
    is_relevant: bool

class JailbreakOutput(BaseModel):
    """Schema for jailbreak guardrail decisions."""
    reasoning: Optional[str]
    is_safe: bool