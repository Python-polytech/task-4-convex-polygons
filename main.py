from __future__ import annotations # для указания типа класса внутри самого класса
from dataclasses import dataclass
 
import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterable, Iterator

EPS = 1e-9

@dataclass(frozen=True) # делаем класс немутабельным по принципу Барбары Лисков
class Point:
    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

# > 0 - b находится против часовой стрелки от луча oa
# < 0 - b находится по часовой стрелке от луча oa
# = 0 - три точки лежат на одной прямой
def vector_product(o: Point, a: Point, b: Point) -> float:
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

# формула Гаусса для нахождения площади многоугольника
def signed_area(points: Iterable[Point]) -> float:
    pts = list(points)
    n = len(pts)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        total += a.x * b.y - b.x * a.y
    return total / 2.0

def line_intersection(p1: Point, p2: Point, a: Point, b: Point) -> Point:
    """Находит точку пересечения двух прямых, заданных парами точек.

    Прямая 1 задана точками `p1` и `p2`, прямая 2 — точками `a` и `b`.
    Использует параметрическое уравнение прямой и векторные произведения.

    Returns:
        Point: Точка пересечения прямых.
    """
    d1 = vector_product(a, b, p1)
    d2 = vector_product(a, b, p2)
    t = d1 / (d1 - d2)
    return Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y))

# согласное принципу единственности ответственности, выделяем отдельные абстрактные классы для триангуляции и пересечения многоугольников
class Triangulator(ABC):
    @abstractmethod
    def triangulate(self, polygon: ConvexPolygon) -> tuple[ConvexPolygon, ...]:
        ...
 
 
class Intersector(ABC):
    @abstractmethod
    def intersect(self, first: ConvexPolygon, second: ConvexPolygon) -> ConvexPolygon:
        ...

class ConvexPolygon:
    def __init__(self, vertices: Iterable[Point]) -> None:
        self._vertices = self._validate_and_normalize(tuple(vertices))

    @classmethod
    def empty(cls) -> ConvexPolygon:
        obj = cls.__new__(cls)
        obj._vertices = ()
        return obj
 
    @classmethod
    def from_tuples(cls, points: Iterable[tuple[float, float]]) -> ConvexPolygon:
        return cls(Point(x, y) for x, y in points)
 
    @classmethod
    def regular(cls, n: int, radius: float, center: Point = Point(0.0, 0.0)) -> ConvexPolygon:
        """Создаёт правильный n-угольник, вписанный в окружность заданного радиуса.

        Returns:
            ConvexPolygon: Новый правильный многоугольник.

        Raises:
            ValueError: Если n < 3 или radius <= 0.
        """
        if n < 3:
            raise ValueError("у многоугольника должно быть не меньше 3 вершин")
        if radius <= 0:
            raise ValueError("радиус должен быть положительным")
        return cls(
            Point(
                center.x + radius * math.cos(2 * math.pi * k / n),
                center.y + radius * math.sin(2 * math.pi * k / n),
            )
            for k in range(n)
        )
    
    @staticmethod
    def _validate_and_normalize(vertices: tuple[Point, ...]) -> tuple[Point, ...]:
        """Проверяет корректность вершин и нормализует порядок обхода.

        Проверяет, что вершин не меньше трёх, что многоугольник выпуклый,
        что нет коллинеарных соседних вершин, и что все вершины лежат
        по одну сторону от каждого ребра. Если обход идёт по часовой
        стрелке, порядок вершин инвертируется.

        Args:
            vertices (tuple[Point, ...]): Кортеж вершин многоугольника.

        Returns:
            tuple[Point, ...]: Нормализованный кортеж вершин (обход против часовой стрелки).

        Raises:
            ValueError: Если вершин меньше 3, они коллинеарны или многоугольник невыпуклый.
        """
        n = len(vertices)
        if n < 3:
            raise ValueError("у многоугольника должно быть не меньше 3 вершин")

        if signed_area(vertices) < 0:
            vertices = vertices[::-1]

        for i in range(n):
            a, b, c = vertices[i], vertices[(i + 1) % n], vertices[(i + 2) % n]
            if vector_product(a, b, c) < EPS:
                raise ValueError("вершины совпадают, коллинеарны или многоугольник невыпуклый")
            if any(vector_product(a, b, p) < -EPS for p in vertices):
                raise ValueError("многоугольник невыпуклый")
        return vertices

    @property
    def vertices(self) -> tuple[Point, ...]:
        return self._vertices

    @property
    def is_empty(self) -> bool:
        return not self._vertices
 
    def __len__(self) -> int:
        return len(self._vertices)
 
    def __iter__(self) -> Iterator[Point]:
        return iter(self._vertices)
 
    def edges(self) -> Iterator[tuple[Point, Point]]:
        """Генерирует рёбра многоугольника как пары соседних вершин.

        Yields:
            tuple[Point, Point]: Пара точек (начало, конец) каждого ребра.
        """
        n = len(self._vertices)
        for i in range(n):
            yield self._vertices[i], self._vertices[(i + 1) % n]
 
    # посколькую многоугольники неизменяемые, можно кэшировать вычисляемые свойства
    @cached_property
    def area(self) -> float:
        return signed_area(self._vertices)
 
    @cached_property
    def perimeter(self) -> float:
        return sum((a.distance_to(b) for a, b in self.edges()), 0.0)

    def contains(self, point: Point) -> bool:
        """Проверяет, лежит ли точка внутри или на границе многоугольника."""
        if self.is_empty:
            return False
        return all(vector_product(a, b, point) >= -EPS for a, b in self.edges())
 
    def __contains__(self, point: object) -> bool:
        return isinstance(point, Point) and self.contains(point)

    def triangulate(self, triangulator: Triangulator) -> tuple[ConvexPolygon, ...]:
        """Разбивает многоугольник на треугольники с помощью указанного алгоритма."""
        return triangulator.triangulate(self)

    def intersect(self, other: ConvexPolygon, intersector: Intersector) -> ConvexPolygon:
        """Находит пересечение с другим выпуклым многоугольником с помощью указанного алгоритма."""
        return intersector.intersect(self, other)
 
    def __repr__(self) -> str:
        pts = ", ".join(f"({v.x:g}, {v.y:g})" for v in self._vertices)
        return f"ConvexPolygon([{pts}])"
    
class FanTriangulator(Triangulator):
    def triangulate(self, polygon: ConvexPolygon) -> tuple[ConvexPolygon, ...]:
        """Выполняет веерную триангуляцию выпуклого многоугольника."""

        v = polygon.vertices
        return tuple(ConvexPolygon((v[0], v[i], v[i + 1])) for i in range(1, len(v) - 1))

class MyIntersector(Intersector):
    def intersect(self, first: ConvexPolygon, second: ConvexPolygon) -> ConvexPolygon:
        """Находит пересечение двух выпуклых многоугольников алгоритмом Сазерленда-Ходжмана.

        Args:
            first (ConvexPolygon): Первый многоугольник (заготовка).
            second (ConvexPolygon): Второй многоугольник (отсекатель).

        Returns:
            ConvexPolygon: Результат пересечения или пустой многоугольник.
        """
        if first.is_empty or second.is_empty:
            return ConvexPolygon.empty()
 
        result = list(first.vertices)
        for a, b in second.edges():
            result = self._clip(result, a, b) # выполняем отсечение первого многоугольника по каждой грани второго многоугольника
 
        cleaned = self._clean(result)
        if len(cleaned) < 3 or signed_area(cleaned) < EPS:
            return ConvexPolygon.empty() # если в пересечении меньше 3 вершин или площадь меньше EPS, возвращаем пустой многоугольник
        return ConvexPolygon(cleaned)
 
    @staticmethod
    def _clip(points: list[Point], a: Point, b: Point) -> list[Point]:
        """Отсекает многоугольник полуплоскостью, заданной прямой (a, b).

        Args:
            points (list[Point]): Вершины текущего многоугольника.
            a (Point): Начало отсекающего ребра.
            b (Point): Конец отсекающего ребра.

        Returns:
            list[Point]: Вершины отсечённого многоугольника.
        """
        if not points:
            return []
 
        result: list[Point] = []
        prev = points[-1]
        for cur in points:
            cur_inside = vector_product(a, b, cur) >= -EPS # > 0 - cur находится против часовой стрелки от луча ab
            prev_inside = vector_product(a, b, prev) >= -EPS # > 0 - prev находится против часовой стрелки от луча ab
            if cur_inside:
                if not prev_inside:
                    result.append(line_intersection(prev, cur, a, b)) # если cur находится внутри, а prev снаружи, добавляем точку пересечения
                result.append(cur)
            elif prev_inside: 
                result.append(line_intersection(prev, cur, a, b)) # если cur находится снаружи, а prev внутри, добавляем точку пересечения
            prev = cur
        return result
 
    @staticmethod
    def _clean(points: list[Point]) -> list[Point]:
        """Удаляет дубликаты и коллинеарные вершины из списка точек.

        Args:
            points (list[Point]): Список вершин многоугольника.

        Returns:
            list[Point]: Очищенный список вершин.
        """
        pts = list(points)
        changed = True
        while changed and len(pts) >= 3:
            changed = False
            for i in range(len(pts)):
                prev, cur, nxt = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
                if cur.distance_to(prev) < EPS or abs(vector_product(prev, cur, nxt)) < EPS: # если вершина совпадает с предыдущей или коллинеарна с соседними, удаляем ее
                    del pts[i]
                    changed = True
                    break # после удаления вершины индесы сдивагются => начинаем цикл заново
        return pts

class ConsoleApp:

    def __init__(self) -> None:
        self.polygons: list[ConvexPolygon] = []
        self.triangulator = FanTriangulator()
        self.intersector = MyIntersector()
        self.actions = {
            "1": ("Создать правильный n-угольник", self.create_regular),
            "2": ("Создать многоугольник из координат", self.create_from_tuples),
            "3": ("Показать список", self.show),
            "4": ("Пересечение", self.intersect),
            "5": ("Триангуляция", self.triangulate),
        }

    def create_regular(self) -> None:
        try:
            n = int(input("Вершин (>=3): "))
            r = float(input("Радиус: "))
            self.polygons.append(ConvexPolygon.regular(n, r))
            print("OK")
        except ValueError as e:
            print(f"Ошибка: {e}")

    def create_from_tuples(self) -> None:
        try:
            raw = input("Координаты (x1 y1 x2 y2 ...): ")
            nums = list(map(float, raw.split()))
            if len(nums) < 6 or len(nums) % 2 != 0:
                raise ValueError("Нужно чётное число координат (минимум 3 точки).")
            points = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
            self.polygons.append(ConvexPolygon.from_tuples(points))
            print("OK")
        except ValueError as e:
            print(f"Ошибка: {e}")

    def show(self) -> None:

        if not self.polygons:
            print("Пусто.")
            return
        for i, p in enumerate(self.polygons):
            print(f"[{i}] {p} | S={p.area:.4f}, P={p.perimeter:.4f}")

    def intersect(self) -> None:
        self.show()
        if len(self.polygons) < 2:
            print("Нужно 2 многоугольника.")
            return
        try:
            i, j = map(int, input("Индексы (i j): ").split())
            res = self.polygons[i].intersect(self.polygons[j], self.intersector)
            print("Пусто." if res.is_empty else f"{res} | S={res.area:.4f}, P={res.perimeter:.4f}")
        except (ValueError, IndexError):
            print("Неверный ввод.")

    def triangulate(self) -> None:
        self.show()
        try:
            idx = int(input("Индекс: "))
            tris = self.polygons[idx].triangulate(self.triangulator)
            print(f"Треугольников: {len(tris)}")
            for t in tris:
                print(f"  {t}")
        except (ValueError, IndexError):
            print("Неверный ввод.")

    def run(self) -> None:
        while True:
            print("\n--- Меню ---")
            for key, (desc, _) in self.actions.items():
                print(f"{key}. {desc}")
            print("0. Выход")
            choice = input("Выбор: ").strip()
            if choice == "0":
                break
            if choice in self.actions:
                self.actions[choice][1]()
            else:
                print("Неизвестный пункт.")


if __name__ == "__main__":
    ConsoleApp().run()