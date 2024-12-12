from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import Column, TIMESTAMP, Integer, String, Float, Boolean


class Base(DeclarativeBase):
    pass


class Price (Base):
    __tablename__ = "prices"
    id: Mapped[int] = mapped_column(ForeignKey("items.item_id"), primary_key=True)
    unix_timestamp: Mapped[int] = mapped_column(primary_key=True)
    price: Mapped[float]
    stacks: Mapped[int]

    def __repr__(self) -> str:
        return f"price(item.id={self.id!r}, unix_timestamp={self.unix_timestamp!r}, price={self.price!r}, stacks={self.stacks!r})"


class Item(Base):
    """
    SQLite-compatible version of the items schema.
    """
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)
    item_id = Column(Integer, nullable=False, default=0)
    parent_id = Column(Integer, ForeignKey("items.id"), nullable=True)
    parent_item_id = Column(Integer, ForeignKey("items.item_id"), nullable=True)
    expansion_id = Column(Integer, nullable=True)
    faction = Column(String(2), nullable=True)  # SQLite uses String instead of VARCHAR
    name = Column(String(255), nullable=False, default="")  # String is SQLite's equivalent
    weight = Column(Float, nullable=True)  # SQLite's Float has no precision/scale options
    quality = Column(Integer, nullable=False, default=0)  # SQLite lacks TINYINT, use Integer
    is_heroic = Column(Boolean, nullable=False, default=False)
    display_id = Column(Integer, nullable=False, default=0)  # SQLite doesn't have MEDIUMINT
    inventory_type = Column(Integer, nullable=False, default=0)  # Integer for TINYINT
    allowable_class = Column(Integer, nullable=False, default=-1)  # Integer for MEDIUMINT
    item_level = Column(Integer, nullable=False, default=0)  # Integer for SMALLINT
    required_level = Column(Integer, nullable=False, default=0)  # Integer for TINYINT
    set_id = Column(Integer, nullable=False, default=0)  # Integer for MEDIUMINT
    name_cn = Column(String(255), nullable=False, default="")
    name_de = Column(String(255), nullable=False, default="")
    name_es = Column(String(255), nullable=False, default="")
    name_fr = Column(String(255), nullable=False, default="")
    name_it = Column(String(255), nullable=False, default="")
    name_ko = Column(String(255), nullable=False, default="")
    name_pt = Column(String(255), nullable=False, default="")
    name_ru = Column(String(255), nullable=False, default="")
    created_at = Column(TIMESTAMP, nullable=True)  # Compatible with SQLite
    updated_at = Column(TIMESTAMP, nullable=True)  # Compatible with SQLite

    def __repr__(self) -> str:
        return f"Item(item.id={self.item_id!r}, name={self.name_de!r})"


class SpellDbc(Base):
    __tablename__ = 'spell_dbc'

    Id = Column(Integer, primary_key=True, autoincrement=False)
    Dispel = Column(Integer, nullable=False, default=0)
    Mechanic = Column(Integer, nullable=False, default=0)
    Attributes = Column(Integer, nullable=False, default=0)
    AttributesEx = Column(Integer, nullable=False, default=0)
    AttributesEx2 = Column(Integer, nullable=False, default=0)
    AttributesEx3 = Column(Integer, nullable=False, default=0)
    AttributesEx4 = Column(Integer, nullable=False, default=0)
    AttributesEx5 = Column(Integer, nullable=False, default=0)
    AttributesEx6 = Column(Integer, nullable=False, default=0)
    AttributesEx7 = Column(Integer, nullable=False, default=0)
    Stances = Column(Integer, nullable=False, default=0)
    StancesNot = Column(Integer, nullable=False, default=0)
    Targets = Column(Integer, nullable=False, default=0)
    CastingTimeIndex = Column(Integer, nullable=False, default=1)
    AuraInterruptFlags = Column(Integer, nullable=False, default=0)
    ProcFlags = Column(Integer, nullable=False, default=0)
    ProcChance = Column(Integer, nullable=False, default=0)
    ProcCharges = Column(Integer, nullable=False, default=0)
    MaxLevel = Column(Integer, nullable=False, default=0)
    BaseLevel = Column(Integer, nullable=False, default=0)
    SpellLevel = Column(Integer, nullable=False, default=0)
    DurationIndex = Column(Integer, nullable=False, default=0)
    RangeIndex = Column(Integer, nullable=False, default=1)
    StackAmount = Column(Integer, nullable=False, default=0)
    EquippedItemClass = Column(Integer, nullable=False, default=-1)
    EquippedItemSubClassMask = Column(Integer, nullable=False, default=0)
    EquippedItemInventoryTypeMask = Column(Integer, nullable=False, default=0)
    Effect1 = Column(Integer, nullable=False, default=0)
    Effect2 = Column(Integer, nullable=False, default=0)
    Effect3 = Column(Integer, nullable=False, default=0)
    EffectDieSides1 = Column(Integer, nullable=False, default=0)
    EffectDieSides2 = Column(Integer, nullable=False, default=0)
    EffectDieSides3 = Column(Integer, nullable=False, default=0)
    EffectRealPointsPerLevel1 = Column(Float, nullable=False, default=0.0)
    EffectRealPointsPerLevel2 = Column(Float, nullable=False, default=0.0)
    EffectRealPointsPerLevel3 = Column(Float, nullable=False, default=0.0)
    EffectBasePoints1 = Column(Integer, nullable=False, default=0)
    EffectBasePoints2 = Column(Integer, nullable=False, default=0)
    EffectBasePoints3 = Column(Integer, nullable=False, default=0)
    EffectMechanic1 = Column(Integer, nullable=False, default=0)
    EffectMechanic2 = Column(Integer, nullable=False, default=0)
    EffectMechanic3 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetA1 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetA2 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetA3 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetB1 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetB2 = Column(Integer, nullable=False, default=0)
    EffectImplicitTargetB3 = Column(Integer, nullable=False, default=0)
    EffectRadiusIndex1 = Column(Integer, nullable=False, default=0)
    EffectRadiusIndex2 = Column(Integer, nullable=False, default=0)
    EffectRadiusIndex3 = Column(Integer, nullable=False, default=0)
    EffectApplyAuraName1 = Column(Integer, nullable=False, default=0)
    EffectApplyAuraName2 = Column(Integer, nullable=False, default=0)
    EffectApplyAuraName3 = Column(Integer, nullable=False, default=0)
    EffectAmplitude1 = Column(Integer, nullable=False, default=0)
    EffectAmplitude2 = Column(Integer, nullable=False, default=0)
    EffectAmplitude3 = Column(Integer, nullable=False, default=0)
    EffectMultipleValue1 = Column(Float, nullable=False, default=0.0)
    EffectMultipleValue2 = Column(Float, nullable=False, default=0.0)
    EffectMultipleValue3 = Column(Float, nullable=False, default=0.0)
    EffectItemType1 = Column(Integer, nullable=False, default=0)
    EffectItemType2 = Column(Integer, nullable=False, default=0)
    EffectItemType3 = Column(Integer, nullable=False, default=0)
    EffectMiscValue1 = Column(Integer, nullable=False, default=0)
    EffectMiscValue2 = Column(Integer, nullable=False, default=0)
    EffectMiscValue3 = Column(Integer, nullable=False, default=0)
    EffectMiscValueB1 = Column(Integer, nullable=False, default=0)
    EffectMiscValueB2 = Column(Integer, nullable=False, default=0)
    EffectMiscValueB3 = Column(Integer, nullable=False, default=0)
    EffectTriggerSpell1 = Column(Integer, nullable=False, default=0)
    EffectTriggerSpell2 = Column(Integer, nullable=False, default=0)
    EffectTriggerSpell3 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskA1 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskA2 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskA3 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskB1 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskB2 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskB3 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskC1 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskC2 = Column(Integer, nullable=False, default=0)
    EffectSpellClassMaskC3 = Column(Integer, nullable=False, default=0)
    SpellName = Column(String(100), default=None)
    MaxTargetLevel = Column(Integer, nullable=False, default=0)
    SpellFamilyName = Column(Integer, nullable=False, default=0)
    SpellFamilyFlags1 = Column(Integer, nullable=False, default=0)
    SpellFamilyFlags2 = Column(Integer, nullable=False, default=0)
    SpellFamilyFlags3 = Column(Integer, nullable=False, default=0)
    MaxAffectedTargets = Column(Integer, nullable=False, default=0)
    DmgClass = Column(Integer, nullable=False, default=0)
    PreventionType = Column(Integer, nullable=False, default=0)
    DmgMultiplier1 = Column(Float, nullable=False, default=0.0)
    DmgMultiplier2 = Column(Float, nullable=False, default=0.0)
    DmgMultiplier3 = Column(Float, nullable=False, default=0.0)
    AreaGroupId = Column(Integer, nullable=False, default=0)
    SchoolMask = Column(Integer, nullable=False, default=0)


class SpellNames(Base):
    __tablename__ = 'spell_names'

    id: Mapped[int] = mapped_column(primary_key=True)
    profession_name: Mapped[str]
    profession_name_de: Mapped[str]
    cooldown: Mapped[int]
    skill: Mapped[int]
    name_de: Mapped[str]
    name_en: Mapped[str]
    name_misc: Mapped[str]
