import unittest

from objects.coordinate import TimeCoordinate as TC
from tests.init_constraints import init_constraints


class TestGetNextPositiveConstraints(unittest.TestCase):
    def test_t_before_all_tcs(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        npc = c.get_next_positive_constraint(TC(3, 5, 0))
        self.assertEqual(npc, TC(3, 5, 1))

    def test_t_equals_previous_tc(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        npc = c.get_next_positive_constraint(TC(3, 5, 1))
        self.assertEqual(npc, TC(3, 2, 5))

    def test_t_between_tcs(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        npc = c.get_next_positive_constraint(TC(3, 5, 2))
        self.assertEqual(npc, TC(3, 2, 5))

    def test_t_at_last_tc(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        npc = c.get_next_positive_constraint(TC(3, 5, 6))
        self.assertEqual(npc, None)

    def test_t_after_last_tc(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        npc = c.get_next_positive_constraint(TC(3, 5, 7))
        self.assertEqual(npc, None)


class TestCanFullfillNextPostitiveConstraints(unittest.TestCase):
    def test_can_exactly_make_it(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        tc = TC(6, 3, 1)
        can_fullfill = c.can_fullfill_next_positive_constraint(tc)
        self.assertEqual(can_fullfill, True)

    def test_can_just_not_make_it(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        tc = TC(6, 3, 2)
        can_fullfill = c.can_fullfill_next_positive_constraint(tc)
        self.assertEqual(can_fullfill, False)

    def test_can_make_it_far_away(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 2), TC(999, 999, 2000)]
        c = init_constraints(positive_constraints=positive_constraints)

        tc = TC(0, 0, 2)
        can_fullfill = c.can_fullfill_next_positive_constraint(tc)
        self.assertEqual(can_fullfill, True)

    def test_can_not_make_it_far_away(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 2), TC(999, 999, 2000)]
        c = init_constraints(positive_constraints=positive_constraints)

        tc = TC(0, 0, 3)
        can_fullfill = c.can_fullfill_next_positive_constraint(tc)
        self.assertEqual(can_fullfill, False)

    def test_has_no_future_tc(self):
        positive_constraints = [TC(3, 5, 1), TC(3, 2, 5), TC(3, 4, 6)]
        c = init_constraints(positive_constraints=positive_constraints)

        tc = TC(999, 999, 6)
        can_fullfill = c.can_fullfill_next_positive_constraint(tc)
        self.assertEqual(can_fullfill, True)


if __name__ == "__main__":
    unittest.main()
