from io import BytesIO, TextIOWrapper

import pandas as pd

from ..flex import FlexExtension


class TableExtension(FlexExtension):
    data_extension = "parquet"

    def __init__(self, header={}, data=None, cls=None):
        super().__init__(header=header, cls=cls)
        self.data = data

    @classmethod
    def _prepare_table(cls, name: str, data: pd.DataFrame):
        bio = BytesIO()
        data.to_parquet(bio, index=True)
        info = cls._get_tarinfo_from_bytesio(name, bio)
        return info, bio

    def _prepare(self, name: str):
        cls = self.__class__
        header_fname = f"{name}/header.json"
        data_fname = f"{name}/data.{cls.data_extension}"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        data_info, data_bio = cls._prepare_table(data_fname, self.data)

        return [(header_info, header_bio), (data_info, data_bio)]

    @classmethod
    def _parse_table(cls, bio: BytesIO):
        b = BytesIO(bio.read())
        data = pd.read_parquet(b)
        return data

    @classmethod
    def _parse(cls, header: dict, members: list):
        bio = members[f"data.{cls.data_extension}"]
        data = cls._parse_table(bio)
        ext = cls(header=header, data=data)
        return ext

    def to_dict(self):
        obj = {"header": self.header, "data": self.data.to_dict(orient="records")}
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        data = pd.DataFrame.from_records(data["data"])
        obj = cls(header, data)
        return obj


class AsciiTableExtension(TableExtension):
    data_extension = "txt"

    @classmethod
    def _prepare_table(cls, name: str, data: pd.DataFrame):
        tio = TextIOWrapper(BytesIO(), "utf-8")
        data.to_csv(tio, index=False)
        bio = tio.detach()
        info = cls._get_tarinfo_from_bytesio(name, bio)
        return info, bio

    @staticmethod
    def _parse_table(bio: BytesIO):
        data = pd.read_csv(bio)
        return data


class JSONTableExtension(TableExtension):
    data_extension = "json"

    @classmethod
    def _prepare_table(cls, name: str, data: pd.DataFrame):
        tio = TextIOWrapper(BytesIO(), "utf-8")
        data.to_json(tio, orient="records")
        bio = tio.detach()
        info = cls._get_tarinfo_from_bytesio(name, bio)
        return info, bio

    @staticmethod
    def _parse_table(bio: BytesIO):
        data = pd.read_json(bio, orient="records")
        return data
