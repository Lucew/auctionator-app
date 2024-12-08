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
