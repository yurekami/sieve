"""
Tests for retail/data_gen.py module.

Tests cover rule generation, discount application logic, and random data generation.
"""

import pytest
from retail.data_gen import (
    RetailRuleEngine,
    CustomerType,
    ItemCategory,
    CartItem,
    CustomerProfile,
    DiscountRule,
)


class TestRetailRuleEngine:
    """Tests for RetailRuleEngine class."""

    def test_default_rule_generation(self):
        """Test that engine generates 30 rules by default."""
        engine = RetailRuleEngine()
        assert len(engine.rules) == 30
        assert all(isinstance(rule, DiscountRule) for rule in engine.rules)

    def test_rule_types_present(self):
        """Test that various rule types are generated."""
        engine = RetailRuleEngine()

        # Check for customer type rules
        customer_type_rules = [
            r for r in engine.rules
            if "customer_type" in r.conditions
        ]
        assert len(customer_type_rules) > 0

        # Check for spend-based rules (uses "min_spend" not "min_total_spend")
        spend_rules = [
            r for r in engine.rules
            if "min_spend" in r.conditions
        ]
        assert len(spend_rules) > 0

        # Check for category-based rules
        category_rules = [
            r for r in engine.rules
            if "category" in r.conditions
        ]
        assert len(category_rules) > 0

        # Check for promo code rules
        promo_rules = [
            r for r in engine.rules
            if "promo_code" in r.conditions
        ]
        assert len(promo_rules) > 0

        # Check for membership rules (uses "membership_years" not "min_membership_years")
        membership_rules = [
            r for r in engine.rules
            if "membership_years" in r.conditions
        ]
        assert len(membership_rules) > 0

    def test_rule_discount_types(self):
        """Test that both percentage and fixed amount discount types exist."""
        engine = RetailRuleEngine()

        percentage_rules = [
            r for r in engine.rules
            if r.discount_type == "percentage"
        ]
        fixed_rules = [
            r for r in engine.rules
            if r.discount_type == "fixed_amount"
        ]

        assert len(percentage_rules) > 0
        assert len(fixed_rules) > 0


class TestRuleApplicability:
    """Tests for check_rule_applies method."""

    def setup_method(self):
        """Create engine and test fixtures."""
        self.engine = RetailRuleEngine()

    def test_customer_type_rule(self):
        """Test rule applies based on customer type."""
        # Find a student discount rule
        student_rule = next(
            (r for r in self.engine.rules
             if r.conditions.get("customer_type") == "student"),
            None
        )

        if student_rule:
            student = CustomerProfile(customer_type=CustomerType.STUDENT)
            regular = CustomerProfile(customer_type=CustomerType.REGULAR)
            cart = [CartItem("Item", ItemCategory.BOOKS, 10.0, 1)]

            # Check if rule has spend requirement (uses "min_spend" not "min_total_spend")
            min_spend = student_rule.conditions.get("min_spend", 0)

            applies_student = self.engine.check_rule_applies(
                student_rule, cart, student, max(10.0, min_spend)
            )
            applies_regular = self.engine.check_rule_applies(
                student_rule, cart, regular, max(10.0, min_spend)
            )

            assert applies_student is True
            assert applies_regular is False

    def test_spend_threshold_rule(self):
        """Test rule applies based on spending threshold."""
        # Create a rule with spend requirement (uses "min_spend" not "min_total_spend")
        rule = DiscountRule(
            rule_id="test_spend",
            description="Test spend rule",
            natural_language="10% off for $50+ purchases",
            conditions={"min_spend": 50.0},
            discount_type="percentage",
            discount_value=10.0
        )

        customer = CustomerProfile(customer_type=CustomerType.REGULAR)
        cart = [CartItem("Item", ItemCategory.BOOKS, 30.0, 1)]

        # Below threshold
        assert self.engine.check_rule_applies(rule, cart, customer, 40.0) is False

        # At threshold
        assert self.engine.check_rule_applies(rule, cart, customer, 50.0) is True

        # Above threshold
        assert self.engine.check_rule_applies(rule, cart, customer, 60.0) is True

    def test_promo_code_rule(self):
        """Test rule applies based on promo code."""
        rule = DiscountRule(
            rule_id="test_promo",
            description="Test promo rule",
            natural_language="$10 off with WELCOME10",
            conditions={"promo_code": "WELCOME10"},
            discount_type="fixed_amount",
            discount_value=10.0
        )

        customer = CustomerProfile(customer_type=CustomerType.REGULAR)
        cart = [CartItem("Item", ItemCategory.BOOKS, 30.0, 1)]

        # With correct promo
        assert self.engine.check_rule_applies(
            rule, cart, customer, 30.0, promo_code="WELCOME10"
        ) is True

        # With wrong promo
        assert self.engine.check_rule_applies(
            rule, cart, customer, 30.0, promo_code="INVALID"
        ) is False

        # No promo
        assert self.engine.check_rule_applies(
            rule, cart, customer, 30.0
        ) is False

    def test_category_rule(self):
        """Test rule applies based on cart containing specific category."""
        # Conditions use string category names, not ItemCategory enums
        rule = DiscountRule(
            rule_id="test_category",
            description="Test category rule",
            natural_language="15% off electronics",
            conditions={"category": "electronics"},
            discount_type="percentage",
            discount_value=15.0
        )

        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Cart with electronics
        cart_with = [
            CartItem("Laptop", ItemCategory.ELECTRONICS, 500.0, 1)
        ]
        assert self.engine.check_rule_applies(
            rule, cart_with, customer, 500.0
        ) is True

        # Cart without electronics
        cart_without = [
            CartItem("Book", ItemCategory.BOOKS, 20.0, 1)
        ]
        assert self.engine.check_rule_applies(
            rule, cart_without, customer, 20.0
        ) is False

    def test_membership_years_rule(self):
        """Test rule applies based on membership duration."""
        # Uses "membership_years" not "min_membership_years"
        rule = DiscountRule(
            rule_id="test_membership",
            description="Test membership rule",
            natural_language="5% off for 3+ year members",
            conditions={"membership_years": 3},
            discount_type="percentage",
            discount_value=5.0
        )

        cart = [CartItem("Item", ItemCategory.BOOKS, 30.0, 1)]

        # New member
        new_member = CustomerProfile(
            customer_type=CustomerType.REGULAR,
            membership_years=1
        )
        assert self.engine.check_rule_applies(
            rule, cart, new_member, 30.0
        ) is False

        # Long-time member
        veteran_member = CustomerProfile(
            customer_type=CustomerType.REGULAR,
            membership_years=5
        )
        assert self.engine.check_rule_applies(
            rule, cart, veteran_member, 30.0
        ) is True


class TestDiscountCalculation:
    """Tests for calculate_final_price method."""

    def setup_method(self):
        """Create engine for tests."""
        self.engine = RetailRuleEngine()

    def test_no_discounts(self):
        """Test price remains unchanged when no rules apply."""
        # Use a scenario where default engine's 30 rules won't match
        cart = [CartItem("Book", ItemCategory.BOOKS, 5.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR, membership_years=0)

        final_price = self.engine.calculate_final_price(cart, customer, 5.0)
        # With 5.0 spend and regular customer with 0 years, no rules should apply
        assert final_price == 5.0

    def test_single_percentage_discount(self):
        """Test single percentage discount application."""
        # Create minimal config that generates no rules, then manually add one rule
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        # Manually add a test rule
        engine.rules = [
            DiscountRule(
                rule_id="test1",
                description="Test percentage",
                natural_language="10% off all purchases",
                conditions={},  # Always applies
                discount_type="percentage",
                discount_value=10.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 100.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        final_price = engine.calculate_final_price(cart, customer, 100.0)
        assert final_price == 90.0

    def test_single_fixed_discount(self):
        """Test single fixed amount discount application."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test2",
                description="Test fixed",
                natural_language="$10 off all purchases",
                conditions={},
                discount_type="fixed_amount",
                discount_value=10.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 50.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        final_price = engine.calculate_final_price(cart, customer, 50.0)
        assert final_price == 40.0

    def test_category_percentage_discount(self):
        """Test category-specific percentage discount."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test3",
                description="Electronics discount",
                natural_language="15% off electronics",
                conditions={"category": "electronics"},
                discount_type="percentage",
                discount_value=15.0
            )
        ]

        cart = [
            CartItem("Laptop", ItemCategory.ELECTRONICS, 100.0, 1),
            CartItem("Book", ItemCategory.BOOKS, 20.0, 1)
        ]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Electronics: 100 * 0.85 = 85
        # Books: 20 (no discount)
        # Total: 105
        final_price = engine.calculate_final_price(cart, customer, 120.0)
        assert final_price == 105.0

    def test_multiple_category_discounts_highest_wins(self):
        """Test that highest category discount is applied per category."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test4a",
                description="Electronics 10%",
                natural_language="10% off electronics",
                conditions={"category": "electronics"},
                discount_type="percentage",
                discount_value=10.0
            ),
            DiscountRule(
                rule_id="test4b",
                description="Electronics 20%",
                natural_language="20% off electronics",
                conditions={"category": "electronics"},
                discount_type="percentage",
                discount_value=20.0
            )
        ]

        cart = [CartItem("Laptop", ItemCategory.ELECTRONICS, 100.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Should apply 20% (highest), not 10%
        final_price = engine.calculate_final_price(cart, customer, 100.0)
        assert final_price == 80.0

    def test_total_percentage_discount_after_category(self):
        """Test total percentage discount applies after category discounts."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test5a",
                description="Electronics 15%",
                natural_language="15% off electronics",
                conditions={"category": "electronics"},
                discount_type="percentage",
                discount_value=15.0
            ),
            DiscountRule(
                rule_id="test5b",
                description="Total 10%",
                natural_language="10% off total",
                conditions={},
                discount_type="percentage",
                discount_value=10.0
            )
        ]

        cart = [
            CartItem("Laptop", ItemCategory.ELECTRONICS, 100.0, 1),
            CartItem("Book", ItemCategory.BOOKS, 20.0, 1)
        ]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Step 1: Category discount on electronics: 100 * 0.85 = 85
        # Running total: 85 + 20 = 105
        # Step 2: Total discount: 105 * 0.90 = 94.5
        final_price = engine.calculate_final_price(cart, customer, 120.0)
        assert final_price == 94.5

    def test_fixed_discount_last(self):
        """Test fixed discounts apply after percentage discounts."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test6a",
                description="10% off",
                natural_language="10% off total",
                conditions={},
                discount_type="percentage",
                discount_value=10.0
            ),
            DiscountRule(
                rule_id="test6b",
                description="$5 off",
                natural_language="$5 off",
                conditions={},
                discount_type="fixed_amount",
                discount_value=5.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 100.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Step 1: 10% off: 100 * 0.90 = 90
        # Step 2: $5 off: 90 - 5 = 85
        final_price = engine.calculate_final_price(cart, customer, 100.0)
        assert final_price == 85.0

    def test_multiple_fixed_discounts_sum(self):
        """Test multiple fixed discounts are summed."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test7a",
                description="$5 off",
                natural_language="$5 off",
                conditions={},
                discount_type="fixed_amount",
                discount_value=5.0
            ),
            DiscountRule(
                rule_id="test7b",
                description="$10 off",
                natural_language="$10 off",
                conditions={},
                discount_type="fixed_amount",
                discount_value=10.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 100.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # Both fixed discounts: 100 - 5 - 10 = 85
        final_price = engine.calculate_final_price(cart, customer, 100.0)
        assert final_price == 85.0

    def test_price_floors_at_zero(self):
        """Test price cannot go below zero."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test8",
                description="$100 off",
                natural_language="$100 off",
                conditions={},
                discount_type="fixed_amount",
                discount_value=100.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 50.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        final_price = engine.calculate_final_price(cart, customer, 50.0)
        assert final_price == 0.0

    def test_promo_code_discount(self):
        """Test promo code discount application."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test9",
                description="Promo WELCOME10",
                natural_language="$10 off with WELCOME10",
                conditions={"promo_code": "WELCOME10"},
                discount_type="fixed_amount",
                discount_value=10.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 50.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # With promo
        final_price = engine.calculate_final_price(
            cart, customer, 50.0, promo_code="WELCOME10"
        )
        assert final_price == 40.0

        # Without promo
        final_price_no_promo = engine.calculate_final_price(
            cart, customer, 50.0
        )
        assert final_price_no_promo == 50.0

    def test_complex_scenario_student_electronics(self):
        """Test complex scenario: student with electronics purchase."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test10a",
                description="Student 10%",
                natural_language="10% off for students on $50+",
                conditions={
                    "customer_type": "student",
                    "min_spend": 50.0
                },
                discount_type="percentage",
                discount_value=10.0
            ),
            DiscountRule(
                rule_id="test10b",
                description="Electronics 15%",
                natural_language="15% off electronics",
                conditions={"category": "electronics"},
                discount_type="percentage",
                discount_value=15.0
            )
        ]

        cart = [CartItem("Laptop", ItemCategory.ELECTRONICS, 60.0, 1)]
        student = CustomerProfile(customer_type=CustomerType.STUDENT)

        # Step 1: Category discount (15%): 60 * 0.85 = 51
        # Step 2: Total discount (10%) on 51: 51 * 0.90 = 45.9
        final_price = engine.calculate_final_price(cart, student, 60.0)
        assert final_price == 45.9

    def test_return_applied_rules(self):
        """Test return_applied_rules parameter."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test11",
                description="Test",
                natural_language="10% off",
                conditions={},
                discount_type="percentage",
                discount_value=10.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 100.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        result = engine.calculate_final_price(
            cart, customer, 100.0, return_applied_rules=True
        )

        assert isinstance(result, tuple)
        final_price, applied_rules = result
        assert final_price == 90.0
        assert len(applied_rules) == 1
        assert applied_rules[0].rule_id == "test11"

    def test_rounding_to_two_decimals(self):
        """Test that final price is rounded to 2 decimal places."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        engine = RetailRuleEngine(rule_config=minimal_config)
        engine.rules = [
            DiscountRule(
                rule_id="test12",
                description="Test",
                natural_language="33% off",
                conditions={},
                discount_type="percentage",
                discount_value=33.0
            )
        ]

        cart = [CartItem("Item", ItemCategory.BOOKS, 10.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        # 10 * 0.67 = 6.7 (already 1 decimal)
        final_price = engine.calculate_final_price(cart, customer, 10.0)
        assert final_price == 6.7

        # Test with value needing rounding
        cart2 = [CartItem("Item", ItemCategory.BOOKS, 10.01, 1)]
        final_price2 = engine.calculate_final_price(cart2, customer, 10.01)
        # 10.01 * 0.67 = 6.7067 -> 6.71
        assert final_price2 == 6.71


class TestRandomGeneration:
    """Tests for random data generation methods."""

    def setup_method(self):
        """Create engine for tests."""
        self.engine = RetailRuleEngine()

    def test_generate_random_cart_structure(self):
        """Test random cart generation returns valid structure."""
        cart, total = self.engine.generate_random_cart()

        assert isinstance(cart, list)
        assert isinstance(total, float)
        assert len(cart) >= 1  # At least one item
        assert len(cart) <= 8  # Max 8 items (per source code line 438)

        for item in cart:
            assert isinstance(item, CartItem)
            assert isinstance(item.name, str)
            assert isinstance(item.category, ItemCategory)
            assert isinstance(item.price, float)
            assert isinstance(item.quantity, int)
            assert item.price > 0
            assert item.quantity > 0

    def test_generate_random_cart_total_correct(self):
        """Test that cart total matches sum of items."""
        cart, total = self.engine.generate_random_cart()

        calculated_total = sum(item.price * item.quantity for item in cart)
        assert abs(total - calculated_total) < 0.01  # Allow float precision

    def test_generate_random_customer_structure(self):
        """Test random customer generation returns valid structure."""
        customer = self.engine.generate_random_customer()

        assert isinstance(customer, CustomerProfile)
        assert isinstance(customer.customer_type, CustomerType)
        assert isinstance(customer.membership_years, int)
        assert customer.membership_years >= 0
        assert customer.membership_years <= 10  # Max is 10 per source code line 516

    def test_generate_random_customer_variety(self):
        """Test that random customer generation produces variety."""
        customers = [self.engine.generate_random_customer() for _ in range(20)]

        # Should have at least 2 different customer types
        customer_types = {c.customer_type for c in customers}
        assert len(customer_types) >= 2

        # Should have variety in membership years
        membership_years = {c.membership_years for c in customers}
        assert len(membership_years) >= 2

    def test_generate_random_cart_variety(self):
        """Test that random cart generation produces variety."""
        carts = [self.engine.generate_random_cart() for _ in range(20)]

        # Should have different totals
        totals = {total for _, total in carts}
        assert len(totals) >= 10  # Most should be different

        # Should have different item counts
        item_counts = {len(cart) for cart, _ in carts}
        assert len(item_counts) >= 2


class TestCartFormatting:
    """Tests for cart formatting and prompt generation."""

    def setup_method(self):
        """Create engine and sample data."""
        self.engine = RetailRuleEngine()
        self.cart = [
            CartItem("Laptop", ItemCategory.ELECTRONICS, 999.99, 1),
            CartItem("Mouse", ItemCategory.ELECTRONICS, 29.99, 2),
            CartItem("Book", ItemCategory.BOOKS, 19.99, 1)
        ]
        self.customer = CustomerProfile(
            customer_type=CustomerType.STUDENT,
            membership_years=2
        )

    def test_format_cart_description_structure(self):
        """Test cart description formatting."""
        description = self.engine.format_cart_description(
            self.cart, self.customer, 1079.96
        )

        assert isinstance(description, str)
        assert "Laptop" in description
        assert "999.99" in description
        # Categories are lowercase in the output
        assert "electronics" in description
        # Customer type is lowercase
        assert "student" in description
        assert "Customer Profile:" in description
        assert "Shopping Cart:" in description

    def test_format_cart_description_quantities(self):
        """Test that quantities are shown correctly."""
        description = self.engine.format_cart_description(
            self.cart, self.customer, 1079.96
        )

        # Mouse has quantity 2 - format is "x 2" with space
        assert "Mouse" in description
        assert "29.99" in description
        assert "x 2" in description

    def test_create_prompt_structure(self):
        """Test prompt creation structure."""
        # create_prompt signature: cart, customer, total_spend, promo_code, rules_to_include, include_rules
        prompt = self.engine.create_prompt(
            self.cart, self.customer, 1079.96
        )

        assert isinstance(prompt, str)
        assert "Calculate" in prompt
        assert "discount" in prompt
        assert "student" in prompt
        # Prompt should include the cart description
        assert "Laptop" in prompt

    def test_create_prompt_with_promo(self):
        """Test prompt includes promo code if provided."""
        prompt = self.engine.create_prompt(
            self.cart, self.customer, 1079.96,
            promo_code="SAVE20"
        )

        assert "SAVE20" in prompt

    def test_create_prompt_with_rules(self):
        """Test prompt includes rules when provided."""
        rules = [
            DiscountRule(
                rule_id="r1",
                description="Student discount",
                natural_language="10% off for students",
                conditions={"customer_type": "student"},
                discount_type="percentage",
                discount_value=10.0
            )
        ]

        prompt = self.engine.create_prompt(
            self.cart, self.customer, 1079.96,
            promo_code=None,
            rules_to_include=rules,
            include_rules=True
        )

        # Natural language description should be in the prompt
        assert "10% off for students" in prompt
        assert "Discount Rules:" in prompt


class TestCalculateApplicableDiscounts:
    """Tests for calculate_applicable_discounts method."""

    def setup_method(self):
        """Create engine with known rules."""
        minimal_config = {
            "customer_types": {},
            "categories": {},
            "membership_tiers": {},
            "promo_codes": {},
            "spend_thresholds": [],
            "rule_types": {
                "customer_spend": {"enabled": False},
                "customer_category": {"enabled": False},
                "promo_codes": {"enabled": False},
                "spend_based": {"enabled": False},
                "bulk_category": {"enabled": False},
                "category_spend": {"enabled": False},
            }
        }

        self.engine = RetailRuleEngine(rule_config=minimal_config)
        self.engine.rules = [
            DiscountRule(
                rule_id="r1",
                description="Student discount",
                natural_language="10% off for students",
                conditions={"customer_type": "student"},
                discount_type="percentage",
                discount_value=10.0
            ),
            DiscountRule(
                rule_id="r2",
                description="High spender",
                natural_language="5% off $100+",
                conditions={"min_spend": 100.0},
                discount_type="percentage",
                discount_value=5.0
            ),
            DiscountRule(
                rule_id="r3",
                description="Promo code",
                natural_language="$10 off with SAVE10",
                conditions={"promo_code": "SAVE10"},
                discount_type="fixed_amount",
                discount_value=10.0
            )
        ]

    def test_returns_empty_when_no_rules_apply(self):
        """Test returns empty list when no rules apply."""
        cart = [CartItem("Book", ItemCategory.BOOKS, 10.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        applicable = self.engine.calculate_applicable_discounts(
            cart, customer, 10.0
        )

        assert applicable == []

    def test_returns_single_applicable_rule(self):
        """Test returns single applicable rule."""
        cart = [CartItem("Book", ItemCategory.BOOKS, 20.0, 1)]
        student = CustomerProfile(customer_type=CustomerType.STUDENT)

        applicable = self.engine.calculate_applicable_discounts(
            cart, student, 20.0
        )

        assert len(applicable) == 1
        assert applicable[0].rule_id == "r1"

    def test_returns_multiple_applicable_rules(self):
        """Test returns multiple applicable rules."""
        cart = [CartItem("Laptop", ItemCategory.ELECTRONICS, 150.0, 1)]
        student = CustomerProfile(customer_type=CustomerType.STUDENT)

        applicable = self.engine.calculate_applicable_discounts(
            cart, student, 150.0
        )

        # Both student and high spender rules should apply
        rule_ids = {r.rule_id for r in applicable}
        assert "r1" in rule_ids
        assert "r2" in rule_ids

    def test_promo_code_included(self):
        """Test promo code rule included when provided."""
        cart = [CartItem("Book", ItemCategory.BOOKS, 50.0, 1)]
        customer = CustomerProfile(customer_type=CustomerType.REGULAR)

        applicable = self.engine.calculate_applicable_discounts(
            cart, customer, 50.0, promo_code="SAVE10"
        )

        rule_ids = {r.rule_id for r in applicable}
        assert "r3" in rule_ids
