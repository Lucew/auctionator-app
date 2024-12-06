from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class Item(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    def __repr__(self) -> str:
        return f"Item(id={self.id!r}, name={self.name!r}"


class Price (Base):
    __tablename__ = "prices"
    id: Mapped[int] = mapped_column(ForeignKey("items.id"), primary_key=True)
    unix_timestamp: Mapped[int] = mapped_column(primary_key=True)
    price: Mapped[float]
    stacks: Mapped[int]

    def __repr__(self) -> str:
        return f"price(item.id={self.id!r}, unix_timestamp={self.unix_timestamp!r}, price={self.price!r}, stacks={self.stacks!r})"