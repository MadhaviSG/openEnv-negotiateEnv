"""Twenty B2B SaaS negotiation scenarios with realistic 2025/2026 pricing."""

SCENARIOS = [
    {
        "id": "hubspot_sales_hub_50_seats",
        "product": "HubSpot Sales Hub Professional, 50 seats",
        "context": (
            "You are the procurement manager for a 120-person B2B services firm. You need a modern "
            "CRM to replace spreadsheets. Sales leadership has committed to HubSpot if the price is "
            "right. Your CFO has set a hard cap; you must stay within budget and prefer a shorter commitment."
        ),
        "agent_max_price": 78.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 95.0,
        "vendor_floor_price": 72.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Thanks for your time. For Sales Hub Professional at 50 seats we're at $95 per seat "
            "per month on an annual plan. Our best terms include a three-year commitment with a 7% "
            "annual cap. What timeline are you looking at?"
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "Budget cut 10% — CFO reduced approved spend.",
        "drift_turn": 3,
        "source": "HubSpot 2025 list pricing",
    },
    {
        "id": "salesforce_sales_cloud_100_seats",
        "product": "Salesforce Sales Cloud Enterprise, 100 seats",
        "context": (
            "You represent a mid-market SaaS company scaling sales. Salesforce is the standard in "
            "your space. You need enterprise features but have a strict per-seat budget. Prefer a "
            "2-year deal with a clear cap on increases."
        ),
        "agent_max_price": 148.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 6.0,
        "vendor_list_price": 175.0,
        "vendor_floor_price": 138.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 5.0,
        "vendor_opening_message": (
            "Sales Cloud Enterprise for 100 users is $175 per user per month. We typically do "
            "three-year agreements with a 7% annual cap. I can check on flexibility if you're "
            "ready to move quickly."
        ),
        "opponent_strategy": "hardball",
        "drift_event": "Board requested cost reduction across all vendors.",
        "drift_turn": 4,
        "source": "Salesforce 2025 enterprise pricing",
    },
    {
        "id": "slack_business_plus_200_seats",
        "product": "Slack Business+, 200 seats",
        "context": (
            "You are IT procurement for a distributed tech company. Slack is already in use; you "
            "are formalizing a Business+ deal for compliance and SSO. Budget is fixed for this "
            "fiscal year."
        ),
        "agent_max_price": 13.50,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 15.0,
        "vendor_floor_price": 12.0,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 6.0,
        "vendor_min_cap": 3.0,
        "vendor_opening_message": (
            "Business+ at 200 seats is $15 per user per month on an annual basis. We have a "
            "two-year option at the same rate with a 5% cap. Happy to walk through the "
            "compliance features."
        ),
        "opponent_strategy": "cooperative",
        "drift_event": "New security policy requires signing by end of quarter.",
        "drift_turn": 2,
        "source": "Slack 2025 list pricing",
    },
    {
        "id": "zoom_enterprise_500_seats",
        "product": "Zoom Enterprise, 500 seats",
        "context": (
            "You lead procurement for a global consultancy. Zoom is the preferred video platform. "
            "You need enterprise recording and SSO. Budget was approved at a specific per-seat ceiling."
        ),
        "agent_max_price": 18.0,
        "agent_max_length": 3.0,
        "agent_max_cap": 4.0,
        "vendor_list_price": 22.0,
        "vendor_floor_price": 16.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 5.0,
        "vendor_min_cap": 3.0,
        "vendor_opening_message": (
            "Zoom Enterprise for 500 licenses is $22 per license per month. Our standard is a "
            "three-year agreement with a 5% annual cap. We have limited slots at this tier "
            "this quarter."
        ),
        "opponent_strategy": "urgency",
        "drift_event": "Finance moved the renewal deadline up by 30 days.",
        "drift_turn": 3,
        "source": "Zoom 2025 enterprise pricing",
    },
    {
        "id": "notion_team_75_seats",
        "product": "Notion Team plan, 75 seats",
        "context": (
            "You are the operations lead for a product team. Notion is chosen for wikis and "
            "project tracking. You have a fixed tooling budget and want to avoid long lock-in."
        ),
        "agent_max_price": 14.0,
        "agent_max_length": 1.0,
        "agent_max_cap": 6.0,
        "vendor_list_price": 18.0,
        "vendor_floor_price": 12.0,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 6.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Notion Team for 75 members is $18 per member per month billed annually. We offer "
            "one- and two-year terms; the two-year comes with a slightly better rate. "
            "Annual cap is typically 6%."
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "Headcount increased; need to keep per-seat cost flat.",
        "drift_turn": 2,
        "source": "Notion 2025 team pricing",
    },
    {
        "id": "figma_organization_40_seats",
        "product": "Figma Organization, 40 full seats",
        "context": (
            "You are the design ops lead. Figma Organization is required for design systems and "
            "dev handoff. Budget is approved per seat with a strict cap on annual increases."
        ),
        "agent_max_price": 48.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 55.0,
        "vendor_floor_price": 44.0,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Figma Organization at 40 full seats is $55 per seat per month on an annual plan. "
            "We usually do two-year deals with a 5% cap. I can see what's possible if you're "
            "ready to commit."
        ),
        "opponent_strategy": "cooperative",
        "drift_event": "Design team requested 10 more seats; budget unchanged.",
        "drift_turn": 3,
        "source": "Figma 2025 org pricing",
    },
    {
        "id": "zendesk_suite_growth_60_seats",
        "product": "Zendesk Suite Growth, 60 agent seats",
        "context": (
            "You are the customer success lead procuring a support platform. Zendesk Suite Growth "
            "fits your channel mix. You have a per-agent budget and want a moderate term length."
        ),
        "agent_max_price": 65.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 6.0,
        "vendor_list_price": 79.0,
        "vendor_floor_price": 58.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 8.0,
        "vendor_min_cap": 5.0,
        "vendor_opening_message": (
            "Suite Growth for 60 agents is $79 per agent per month. Our best pricing is on a "
            "three-year commitment with a 6% annual cap. What's your timeline for going live?"
        ),
        "opponent_strategy": "hardball",
        "drift_event": "Support headcount freeze; must keep total cost flat.",
        "drift_turn": 4,
        "source": "Zendesk 2025 suite pricing",
    },
    {
        "id": "intercom_flex_35_seats",
        "product": "Intercom Flex, 35 seats",
        "context": (
            "You are the head of growth for a D2C brand. Intercom Flex is shortlisted for chat "
            "and automation. You need to stay within the approved tools budget and prefer a "
            "two-year deal."
        ),
        "agent_max_price": 99.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 119.0,
        "vendor_floor_price": 89.0,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Intercom Flex at 35 seats runs $119 per seat per month on an annual contract. "
            "We can do a two-year with a 5% cap. We have a quarter-end incentive if you can "
            "sign by Friday."
        ),
        "opponent_strategy": "urgency",
        "drift_event": "Marketing requested additional product; budget reallocated.",
        "drift_turn": 3,
        "source": "Intercom 2025 flex pricing",
    },
    {
        "id": "workday_hcm_mid_400_employees",
        "product": "Workday HCM (mid-market), 400 employees",
        "context": (
            "You are the CHRO's procurement lead. Workday is selected for HCM. You have a "
            "per-employee budget and need a clear cap on annual price increases for planning."
        ),
        "agent_max_price": 11.0,
        "agent_max_length": 3.0,
        "agent_max_cap": 4.0,
        "vendor_list_price": 14.0,
        "vendor_floor_price": 9.5,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 6.0,
        "vendor_min_cap": 3.0,
        "vendor_opening_message": (
            "Workday HCM for 400 employees is $14 per employee per month. We typically sign "
            "three-year agreements with a 4% annual cap. There's limited flexibility below "
            "list for this segment."
        ),
        "opponent_strategy": "hardball",
        "drift_event": "HR requested adding Learning; budget must stay within current envelope.",
        "drift_turn": 5,
        "source": "Workday 2025 mid-market benchmarks",
    },
    {
        "id": "servicenow_itsm_250_seats",
        "product": "ServiceNow IT Service Management, 250 seats",
        "context": (
            "You are the IT director procuring an ITSM platform. ServiceNow is the chosen vendor. "
            "You have a per-seat ceiling and prefer a two-year initial term with a reasonable cap."
        ),
        "agent_max_price": 95.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 115.0,
        "vendor_floor_price": 85.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "ServiceNow ITSM for 250 users is $115 per user per month. Our standard is a "
            "three-year deal with a 5% cap. I'd need to involve my manager for anything "
            "below list."
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "Consolidation initiative: all new tooling must show 15% savings vs. prior year.",
        "drift_turn": 4,
        "source": "ServiceNow 2025 list pricing",
    },
    {
        "id": "hubspot_marketing_hub_30_seats",
        "product": "HubSpot Marketing Hub Professional, 30 seats",
        "context": (
            "You are the demand gen lead. Marketing Hub Professional is selected for campaigns "
            "and attribution. You have a fixed marketing tech budget and want to avoid a long lock-in."
        ),
        "agent_max_price": 24.0,
        "agent_max_length": 1.0,
        "agent_max_cap": 6.0,
        "vendor_list_price": 30.0,
        "vendor_floor_price": 21.5,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Marketing Hub Professional for 30 users runs about $30 per seat per month on an "
            "annual plan. We have a one-year option; the two-year locks in a 6% cap. "
            "What's your budget range?"
        ),
        "opponent_strategy": "urgency",
        "drift_event": "CMO cut martech budget by 15% for this quarter.",
        "drift_turn": 2,
        "source": "HubSpot 2025 marketing hub pricing",
    },
    {
        "id": "salesforce_service_cloud_80_seats",
        "product": "Salesforce Service Cloud Enterprise, 80 seats",
        "context": (
            "You are the support director. Service Cloud is chosen for case management and "
            "knowledge base. You have a per-agent budget and need a deal that fits your fiscal year."
        ),
        "agent_max_price": 155.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 180.0,
        "vendor_floor_price": 142.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Service Cloud Enterprise for 80 agents is $180 per agent per month. Three-year "
            "agreements get our best terms with a 5% cap. I can run numbers for a two-year "
            "if that's your preference."
        ),
        "opponent_strategy": "cooperative",
        "drift_event": "Merger announced; contract must allow for seat count adjustment.",
        "drift_turn": 3,
        "source": "Salesforce 2025 service cloud pricing",
    },
    {
        "id": "slack_enterprise_500_seats",
        "product": "Slack Enterprise Grid, 500 seats",
        "context": (
            "You are the CIO's deputy. Slack Enterprise Grid is required for compliance and "
            "multi-workspace. You have a per-seat cap and want a clear multi-year cap on increases."
        ),
        "agent_max_price": 14.0,
        "agent_max_length": 3.0,
        "agent_max_cap": 4.0,
        "vendor_list_price": 18.0,
        "vendor_floor_price": 12.5,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 6.0,
        "vendor_min_cap": 3.0,
        "vendor_opening_message": (
            "Enterprise Grid at 500 seats is $18 per user per month. We typically do three-year "
            "commitments with a 4% annual cap. There's one slot left at this pricing for "
            "this quarter."
        ),
        "opponent_strategy": "urgency",
        "drift_event": "Security audit requires all comms tools under contract by Q2.",
        "drift_turn": 4,
        "source": "Slack 2025 enterprise pricing",
    },
    {
        "id": "zoom_rooms_50_rooms",
        "product": "Zoom Rooms, 50 rooms",
        "context": (
            "You are facilities and IT procuring meeting room hardware and Zoom Rooms. You have a "
            "per-room budget and prefer a two-year agreement."
        ),
        "agent_max_price": 52.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 64.0,
        "vendor_floor_price": 48.0,
        "vendor_preferred_length": 2.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Zoom Rooms for 50 rooms is $64 per room per month. Two-year deals get a 5% cap. "
            "We have a quarter-end promotion if you can sign by month-end."
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "Five additional rooms added to scope; total budget unchanged.",
        "drift_turn": 3,
        "source": "Zoom 2025 rooms pricing",
    },
    {
        "id": "notion_enterprise_150_seats",
        "product": "Notion Enterprise, 150 seats",
        "context": (
            "You are the head of knowledge management. Notion Enterprise is required for SSO and "
            "audit logs. You have a per-seat budget and want a moderate term."
        ),
        "agent_max_price": 22.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 28.0,
        "vendor_floor_price": 19.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Notion Enterprise for 150 members is $28 per member per month. Our best terms are "
            "on a three-year with a 5% cap. I can look at a two-year if that's a constraint."
        ),
        "opponent_strategy": "hardball",
        "drift_event": "Legal requested stricter data residency; may affect timeline.",
        "drift_turn": 4,
        "source": "Notion 2025 enterprise pricing",
    },
    {
        "id": "figma_enterprise_80_seats",
        "product": "Figma Enterprise, 80 full seats",
        "context": (
            "You are the design director. Figma Enterprise is needed for advanced security and "
            "dedicated support. Budget is set; you want a two-year deal with a clear cap."
        ),
        "agent_max_price": 78.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 90.0,
        "vendor_floor_price": 70.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Figma Enterprise at 80 full seats is $90 per seat per month. We usually do "
            "three-year agreements with a 5% cap. I'd need manager approval to go below list."
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "Acquisition added 20 designers; need to keep per-seat cost down.",
        "drift_turn": 3,
        "source": "Figma 2025 enterprise pricing",
    },
    {
        "id": "zendesk_suite_professional_120_seats",
        "product": "Zendesk Suite Professional, 120 agent seats",
        "context": (
            "You are the VP of customer experience. Zendesk Suite Professional is chosen for "
            "omnichannel and automation. You have a per-agent budget and prefer a two-year term."
        ),
        "agent_max_price": 98.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 6.0,
        "vendor_list_price": 115.0,
        "vendor_floor_price": 88.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 8.0,
        "vendor_min_cap": 5.0,
        "vendor_opening_message": (
            "Suite Professional for 120 agents is $115 per agent per month. Three-year "
            "commitments get our best pricing and a 6% cap. What's driving your timeline?"
        ),
        "opponent_strategy": "cooperative",
        "drift_event": "Support moving to 24/7; need cost certainty for next 18 months.",
        "drift_turn": 2,
        "source": "Zendesk 2025 suite professional pricing",
    },
    {
        "id": "intercom_advanced_60_seats",
        "product": "Intercom Advanced, 60 seats",
        "context": (
            "You are the head of customer success. Intercom Advanced is shortlisted for "
            "automation and product tours. You have a strict per-seat cap and want a "
            "two-year deal."
        ),
        "agent_max_price": 135.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 159.0,
        "vendor_floor_price": 118.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 8.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "Intercom Advanced at 60 seats is $159 per seat per month on an annual plan. "
            "We have quarter-end pricing if you can commit to a three-year with a 5% cap."
        ),
        "opponent_strategy": "urgency",
        "drift_event": "Product requested integration budget; some reallocated from CS tools.",
        "drift_turn": 4,
        "source": "Intercom 2025 advanced pricing",
    },
    {
        "id": "workday_finance_mid_300_users",
        "product": "Workday Financial Management (mid-market), 300 users",
        "context": (
            "You are the finance lead for a growing company. Workday Financial Management is "
            "selected. You have a per-user budget and need a multi-year cap for planning."
        ),
        "agent_max_price": 28.0,
        "agent_max_length": 3.0,
        "agent_max_cap": 4.0,
        "vendor_list_price": 35.0,
        "vendor_floor_price": 25.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 6.0,
        "vendor_min_cap": 3.0,
        "vendor_opening_message": (
            "Workday Financial Management for 300 users is $35 per user per month. We typically "
            "do three-year agreements with a 4% annual cap. There's limited room below list "
            "for this tier."
        ),
        "opponent_strategy": "hardball",
        "drift_event": "Audit requires all finance systems under contract by year-end.",
        "drift_turn": 5,
        "source": "Workday 2025 financials benchmarks",
    },
    {
        "id": "servicenow_hr_200_seats",
        "product": "ServiceNow HR Service Delivery, 200 seats",
        "context": (
            "You are the HRIS director. ServiceNow HR is chosen for employee service and case "
            "management. You have a per-seat budget and prefer a two-year initial term."
        ),
        "agent_max_price": 22.0,
        "agent_max_length": 2.0,
        "agent_max_cap": 5.0,
        "vendor_list_price": 28.0,
        "vendor_floor_price": 19.0,
        "vendor_preferred_length": 3.0,
        "vendor_max_cap": 7.0,
        "vendor_min_cap": 4.0,
        "vendor_opening_message": (
            "ServiceNow HR Service Delivery for 200 users is $28 per user per month. Our best "
            "terms are on a three-year with a 5% cap. I can check on a two-year if that's "
            "your preference."
        ),
        "opponent_strategy": "concession_trader",
        "drift_event": "HR transformation delayed; need flexibility on start date.",
        "drift_turn": 3,
        "source": "ServiceNow 2025 HR pricing",
    },
]
