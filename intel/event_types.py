# -*- coding: utf-8 -*-
"""事件类型受控枚举 v2 + 自由词映射表（04 §5.2：受控枚举 + 原始类型词保留）。

原则：
- 枚举扩充到 16 类（覆盖企业情报高频：财报/销量/投融资/出海/诉讼/事故/获奖/政策/扩张）；
- 模型输出的自由类型词经 TYPE_MAP 映射到最近枚举，映射不到归 other；
- 原始词保留在 event_mention.event_type_raw（类型表有版本，可追溯）；
- 永不因类型词不在枚举而丢弃提及（那是证据校验该管的事）。
"""
from __future__ import annotations

EVENT_TYPES_V2 = [
    "product_launch",      # 产品发布
    "price_change",        # 价格调整
    "patent_publication",  # 专利公开
    "merger_deal",         # 并购交易
    "partnership",         # 合作签约
    "executive_change",    # 人事变动
    "financial_report",    # 财报/业绩发布
    "sales_report",        # 销量/交付快报
    "investment",          # 投融资/资本运作
    "market_entry",        # 市场进入/出海
    "legal_action",        # 诉讼/监管/合规
    "incident",            # 事故/召回/负面
    "award",               # 获奖/荣誉/认证
    "policy_change",       # 战略/政策调整
    "expansion",           # 产能/工厂/网络扩张
    "other",
]

TYPE_LABELS = {
    "product_launch": "产品发布", "price_change": "价格调整", "patent_publication": "专利公开",
    "merger_deal": "并购交易", "partnership": "合作签约", "executive_change": "人事变动",
    "financial_report": "财报发布", "sales_report": "销量快报", "investment": "投融资",
    "market_entry": "市场进入", "legal_action": "诉讼监管", "incident": "事故负面",
    "award": "获奖荣誉", "policy_change": "战略调整", "expansion": "产能扩张", "other": "其他",
}

# 自由词 → 枚举（高频映射；未列出的词若包含枚举名子串也按前缀/包含匹配兜底）
TYPE_MAP = {
    # 财报/业绩
    "financial_report": "financial_report", "financial_result": "financial_report",
    "financial_forecast": "financial_report", "financial_performance": "financial_report",
    "financial_impact": "financial_report", "profit_report": "financial_report",
    "profit_decline": "financial_report", "profit_growth": "financial_report",
    "profitability": "financial_report", "earnings": "financial_report",
    "annual_report": "financial_report", "shareholders_meeting": "financial_report",
    "dividend_proposal": "financial_report", "valuation_change": "financial_report",
    "stock_movement": "financial_report", "stock_price_change": "financial_report",
    "repurchase": "investment", "share_repurchase": "investment", "buyback": "investment",
    # 销量/交付
    "sales_report": "sales_report", "sales_performance": "sales_report",
    "sales": "sales_report", "sales_announcement": "sales_report", "sales_update": "sales_report",
    "sales_growth": "sales_report", "sales_increase": "sales_report", "sales_reach": "sales_report",
    "sales_rank": "sales_report", "sales_revenue": "sales_report", "sales_strategy": "other",
    "export_sales": "sales_report", "product_sales": "sales_report", "product_delivery": "sales_report",
    "delivery": "sales_report", "order_receiving": "sales_report", "market_share": "sales_report",
    "market_share_report": "sales_report", "market_position": "sales_report",
    "rank_change": "sales_report", "record_creation": "sales_report",
    "product_ranking": "sales_report", "market_data_release": "other",
    # 投融资/资本
    "investment": "investment", "strategic_investment": "investment",
    "company_investment": "investment", "project_investment": "investment",
    "project_funding": "investment", "funding": "investment", "funding_round": "investment",
    "capital_increase": "investment", "capital_raising": "investment",
    "equity_change": "investment", "ownership_change": "investment", "ipo": "investment",
    "bond_issuance": "investment", "bond_registration": "investment",
    "stock_purchase": "investment", "stock_reorganization": "investment",
    "asset_disposal": "investment", "company_sale": "investment",
    # 出海/市场
    "market_entry": "market_entry", "market_expansion": "market_entry",
    "business_expansion": "expansion", "market_growth": "market_entry",
    "export": "market_entry", "product_export": "market_entry",
    "overseas_expansion": "market_entry", "store_opening": "expansion",
    "location_opening": "expansion", "network_expansion": "expansion",
    "channel_expansion": "market_entry",
    # 扩张/建设
    "expansion": "expansion", "capacity_expansion": "expansion",
    "production_capacity_increase": "expansion", "infrastructure": "expansion",
    "infrastructure_deployment": "expansion", "infrastructure_development": "expansion",
    "infrastructure_upgrade": "expansion", "project_completion": "expansion",
    "project_start": "expansion", "project_launch": "product_launch",
    "project_announcement": "other", "project_progress": "other",
    "project_construction": "expansion", "project_development": "other",
    "project_expansion": "expansion", "mining_license_obtained": "expansion",
    # 诉讼/监管
    "legal_action": "legal_action", "legal_case": "legal_action",
    "legal_litigation": "legal_action", "lawsuit": "legal_action",
    "penalty": "legal_action", "fine": "legal_action", "regulatory_action": "legal_action",
    "compliance": "legal_action", "license_granted": "legal_action",
    "license_revoked": "legal_action", "investigation": "legal_action",
    # 事故/负面
    "incident": "incident", "fire": "incident", "fire_incident": "incident",
    "incident_response": "incident", "product_recall": "incident", "recall": "incident",
    "service_disruption": "incident", "system_failure": "incident",
    "business_closure": "incident", "product_shutdown": "incident",
    "product_withdrawal": "incident", "service_suspension": "incident",
    "product_phase_out": "incident",
    # 获奖/荣誉
    "award": "award", "award_announcement": "award", "award_receiving": "award",
    "award_reception": "award", "award_recognition": "award", "esg_recognition": "award",
    "certification": "award", "product_certification": "award", "rating": "award",
    "contract_award": "award", "project_award": "award", "contest_announcement": "award",
    "membership": "award", "standard_proposal": "award",
    # 战略/政策/组织
    "policy_change": "policy_change", "policy_announcement": "policy_change",
    "policy_update": "policy_change", "policy_launch": "policy_change",
    "strategic_announcement": "policy_change", "strategic_cooperation": "partnership",
    "strategic_partnership": "partnership", "rebranding": "policy_change",
    "brand_rebranding": "policy_change", "company_rebranding": "policy_change",
    "company_reorganization": "policy_change", "organizational_change": "executive_change",
    "company_split": "merger_deal", "company_foundation": "policy_change",
    "company_founding": "policy_change", "company_founder": "executive_change",
    "company_launch": "product_launch", "company_change": "policy_change",
    "company_action": "other", "company_cooperation": "partnership",
    "business_deal": "partnership", "business_growth": "other",
    "carbon_emission_reduction": "policy_change", "donation": "other",
    "product_donation": "other", "esg": "policy_change",
    # 产品/技术
    "software_release": "product_launch", "software_update": "product_launch",
    "feature_release": "product_launch", "feature_addition": "product_launch",
    "feature_update": "product_launch", "feature_announcement": "product_launch",
    "product_update": "product_launch", "product_upgrade": "product_launch",
    "product_application": "product_launch", "product_approval": "product_launch",
    "product_launch_event": "product_launch", "technology_release": "product_launch",
    "technology_launch": "product_launch", "technology_announcement": "product_launch",
    "technology_update": "product_launch", "technology_upgrade": "product_launch",
    "technology_deployment": "product_launch", "technology_integration": "product_launch",
    "technology_innovation": "product_launch", "technology_application": "product_launch",
    "research_and_development": "product_launch", "research_announcement": "product_launch",
    "product_testing": "product_launch", "product_disassembly": "other",
    "product_performance": "other", "product_service": "other",
    "product_warranty_extension": "policy_change", "software": "product_launch",
    # 合作/签约
    "contract_signing": "partnership", "partnership_agreement": "partnership",
    "cooperation": "partnership", "collaboration": "partnership",
    "project_signing": "partnership", "sponsorship": "partnership",
    "project_sponsorship": "partnership",
    # 人事
    "executive_announcement": "executive_change", "executive_statement": "other",
    "board_meeting": "executive_change", "speech": "other", "public_comment": "other",
    "expert_sharing": "other",
    # 活动/会议/其他
    "event": "other", "event_activity": "other", "event_announcement": "other",
    "event_participation": "other", "event_organization": "other", "event_launch": "other",
    "event_scheduling": "other", "event_start": "other", "meeting": "other",
    "exhibition_participation": "other", "training_event": "other",
    "market_analysis": "other", "market_position_change": "other",
    "response": "other", "service_change": "other", "service_launch": "product_launch",
    "service_provision": "other", "service_recovery": "other", "service_resumption": "other",
    "service_center_launch": "expansion", "publication": "other", "report_release": "other",
    "promotion": "other", "milestone_reached": "other", "planned": "other",
    "correction": "other", "administrative_action": "other", "auction": "other",
    "mining_operation_paused": "incident", "organization_creation": "policy_change",
    "organization_foundation": "policy_change", "upgrade": "product_launch",
    "growth": "other",
}

# 事件类型枚举版本（类型表有版本，04 §5.2）
TYPE_SCHEMA_VERSION = "event-type-v2"


def normalize_event_type(raw) -> tuple[str, str]:
    """自由类型词 → (受控枚举, 原始词)。永不为类型词返回拒绝。"""
    if not raw or not isinstance(raw, str):
        return "other", str(raw or "")
    r = raw.strip().lower()
    if r in EVENT_TYPES_V2:
        return r, raw
    if r in TYPE_MAP:
        return TYPE_MAP[r], raw
    # 包含匹配兜底（如 product_launch_2024）
    for k, v in TYPE_MAP.items():
        if k in r:
            return v, raw
    for t in EVENT_TYPES_V2:
        if t in r:
            return t, raw
    return "other", raw
