import random
import pandas as pd
from datasets import load_dataset
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

random.seed(42)


class CustomerType(Enum):
    STUDENT = "student"
    SENIOR = "senior"
    REGULAR = "regular"
    VETERAN = "veteran"
    EMPLOYEE = "employee"
    TEACHER = "teacher"


class ItemCategory(Enum):
    ELECTRONICS = "electronics"
    CLOTHING = "clothing"
    BOOKS = "books"
    FOOD = "food"
    HOME = "home"
    SPORTS = "sports"
    BEAUTY = "beauty"
    HEALTH = "health"


@dataclass
class CartItem:
    name: str
    category: ItemCategory
    price: float
    quantity: int


@dataclass
class CustomerProfile:
    customer_type: CustomerType
    membership_years: int = 0


@dataclass
class DiscountRule:
    rule_id: str
    description: str
    natural_language: str
    conditions: Dict[str, Any]
    discount_type: str  # "percentage", "fixed_amount"
    discount_value: float


class RetailRuleEngine:
    def __init__(self, rule_config=None):
        self.rule_config = rule_config or self._get_default_config()
        self.rules = self._generate_rules()

    def _get_default_config(self):
        """Define the high-level rule configuration that drives automatic rule generation."""
        return {
            "customer_types": {
                "student": {"display_name": "student", "base_discount": 10},
                "senior": {"display_name": "senior citizen", "base_discount": 15},
                "veteran": {"display_name": "veteran"},
                "employee": {"display_name": "employee", "base_discount": 20},
                "teacher": {"display_name": "teacher", "base_discount": 10},
                "regular": {"display_name": "regular"},
            },
            "categories": {
                "electronics": {"price_range": (50, 200)},
                "clothing": {"price_range": (10, 50)},
                "books": {"price_range": (10, 30)},
                "food": {"price_range": (5, 20)},
                "home": {"price_range": (10, 40)},
                "sports": {"price_range": (20, 60)},
                "beauty": {"price_range": (10, 35)},
                "health": {"price_range": (10, 50)},
            },
            "membership_tiers": {
                "bronze": {"min_years": 1, "discount_boost": 5},
                "silver": {"min_years": 3, "discount_boost": 10},
                "gold": {"min_years": 5, "discount_boost": 15},
            },
            "promo_codes": {
                "SAVE20": {
                    "discount_type": "percentage",
                    "discount_value": 20,
                    "min_spend": 100,
                },
                "WELCOME10": {"discount_type": "fixed_amount", "discount_value": 10},
                "STUDENT15": {
                    "discount_type": "percentage",
                    "discount_value": 15,
                    "customer_type": "student",
                },
                "HOLIDAY30": {
                    "discount_type": "percentage",
                    "discount_value": 30,
                    "min_spend": 150,
                },
                "NEWBIE5": {"discount_type": "fixed_amount", "discount_value": 5},
                "TEACHER10": {
                    "discount_type": "percentage",
                    "discount_value": 10,
                    "customer_type": "teacher",
                },
                "BULK10": {
                    "discount_type": "percentage",
                    "discount_value": 10,
                    "min_spend": 200,
                },
            },
            "spend_thresholds": [50, 100, 150, 200, 300, 500],
            "rule_types": {
                "customer_spend": {
                    "enabled": True,
                    "thresholds": [50],
                    "discount_multiplier": 1.0,  # multiply base_discount by this
                },
                "customer_category": {
                    "enabled": True,
                    "discount_boost": 5,  # add this to base_discount
                    "combinations": [
                        ("student", "electronics"),
                        ("student", "books"),
                        ("senior", "food"),
                        ("veteran", "electronics"),
                        ("employee", "home"),
                        ("teacher", "books"),
                    ],
                },
                "promo_codes": {"enabled": True},
                "spend_based": {
                    "enabled": True,
                    "thresholds": [150, 200],
                    "discount_rates": [5, 10],
                },
                "bulk_category": {"enabled": True, "discount_rate": 10},
                "category_spend": {
                    "enabled": True,
                    "spend_multiplier": 1.5,  # category spend threshold = category price_range max * this
                    "discount_rate": 10,
                },
            },
        }

    def _generate_nl_description(
        self, conditions: Dict[str, Any], discount_type: str, discount_value: float
    ) -> str:
        """Generate natural language description from rule conditions using config."""
        parts = []

        # Customer type conditions
        if "customer_type" in conditions:
            customer_type = conditions["customer_type"]
            display_name = self.rule_config["customer_types"][customer_type][
                "display_name"
            ]
            parts.append(f"customer is a {display_name}")

        # Membership years condition
        if "membership_years" in conditions:
            parts.append(
                f"customer has been a member for {conditions['membership_years']} or more years"
            )

        # Promo code conditions
        if "promo_code" in conditions:
            parts.append(f"promo code is '{conditions['promo_code']}'")

        # Spend conditions
        if "min_spend" in conditions:
            parts.append(f"total spend is at least ${conditions['min_spend']:.0f}")

        # Category conditions
        if "category" in conditions:
            category = conditions["category"]
            if "min_quantity" in conditions:
                parts.append(
                    f"cart contains {conditions['min_quantity']} or more {category} items"
                )
            elif "category_min_spend" in conditions:
                parts.append(
                    f"cart contains {category} items AND total {category} spend is ${conditions['category_min_spend']:.0f} or greater"
                )
            else:
                parts.append(f"cart contains {category}")

        # Join conditions with AND
        condition_text = " AND ".join(parts) if parts else "always"

        # Generate action text
        if discount_type == "percentage":
            if "category" in conditions and "min_quantity" not in conditions:
                action = f"apply {discount_value:.0f}% discount on {conditions['category']} items only"
            else:
                action = f"apply {discount_value:.0f}% discount to total purchase"
        elif discount_type == "fixed_amount":
            action = f"apply ${discount_value:.0f} fixed discount"
        else:
            action = f"apply {discount_type} discount"

        return f"If {condition_text}, {action}"

    def _generate_rules(self) -> List[DiscountRule]:
        """Generate rules automatically from high-level configuration."""
        rules = []
        config = self.rule_config

        # 1. Customer type + spend threshold rules
        for customer_type, customer_config in config["customer_types"].items():
            # Only create rules for customer types that have base_discount defined
            if "base_discount" in customer_config:
                for threshold in config["rule_types"]["customer_spend"]["thresholds"]:
                    discount_pct = (
                        customer_config["base_discount"]
                        * config["rule_types"]["customer_spend"]["discount_multiplier"]
                    )
                    conditions = {
                        "customer_type": customer_type,
                        "min_spend": float(threshold),
                    }
                    rule_id = f"{customer_type}_{threshold}"

                    rules.append(
                        DiscountRule(
                            rule_id=rule_id,
                            description=f"{customer_config['display_name'].title()} discount for purchases over ${threshold}",
                            natural_language=self._generate_nl_description(
                                conditions, "percentage", discount_pct
                            ),
                            conditions=conditions,
                            discount_type="percentage",
                            discount_value=float(discount_pct),
                        )
                    )

        # 2. Customer type + category rules
        if config["rule_types"]["customer_category"]["enabled"]:
            for customer_type, category in config["rule_types"]["customer_category"][
                "combinations"
            ]:
                if (
                    customer_type in config["customer_types"]
                    and category in config["categories"]
                ):
                    # Use base_discount if available, otherwise use the discount_boost as the base discount
                    base_discount = config["customer_types"][customer_type].get(
                        "base_discount", 0
                    )
                    discount_pct = (
                        base_discount
                        + config["rule_types"]["customer_category"]["discount_boost"]
                    )
                    conditions = {"customer_type": customer_type, "category": category}
                    rule_id = f"{customer_type}_{category}"

                    rules.append(
                        DiscountRule(
                            rule_id=rule_id,
                            description=f"{config['customer_types'][customer_type]['display_name'].title()} {category} discount",
                            natural_language=self._generate_nl_description(
                                conditions, "percentage", discount_pct
                            ),
                            conditions=conditions,
                            discount_type="percentage",
                            discount_value=float(discount_pct),
                        )
                    )

        # 3. Promo code rules
        if config["rule_types"]["promo_codes"]["enabled"]:
            for promo_code, promo_config in config["promo_codes"].items():
                conditions = {"promo_code": promo_code}

                # Add additional conditions from promo config
                if "min_spend" in promo_config:
                    conditions["min_spend"] = promo_config["min_spend"]
                if "customer_type" in promo_config:
                    conditions["customer_type"] = promo_config["customer_type"]

                rule_id = promo_code.lower()

                rules.append(
                    DiscountRule(
                        rule_id=rule_id,
                        description=f"{promo_code} promo code discount",
                        natural_language=self._generate_nl_description(
                            conditions,
                            promo_config["discount_type"],
                            promo_config["discount_value"],
                        ),
                        conditions=conditions,
                        discount_type=promo_config["discount_type"],
                        discount_value=float(promo_config["discount_value"]),
                    )
                )

        # 4. Spend-based rules
        if config["rule_types"]["spend_based"]["enabled"]:
            thresholds = config["rule_types"]["spend_based"]["thresholds"]
            discount_rates = config["rule_types"]["spend_based"]["discount_rates"]

            for threshold, discount_pct in zip(thresholds, discount_rates):
                conditions = {"min_spend": float(threshold)}
                rule_id = f"spend_{threshold}"

                rules.append(
                    DiscountRule(
                        rule_id=rule_id,
                        description=f"High spend discount (${threshold}+)",
                        natural_language=self._generate_nl_description(
                            conditions, "percentage", discount_pct
                        ),
                        conditions=conditions,
                        discount_type="percentage",
                        discount_value=float(discount_pct),
                    )
                )

        # 5. Bulk category rules
        if config["rule_types"]["bulk_category"]["enabled"]:
            for category, category_config in config["categories"].items():
                # Only create bulk rules for categories that have bulk_threshold defined
                if "bulk_threshold" in category_config:
                    min_qty = category_config["bulk_threshold"]
                    discount_pct = config["rule_types"]["bulk_category"][
                        "discount_rate"
                    ]
                    conditions = {"category": category, "min_quantity": min_qty}
                    rule_id = f"bulk_{category}"

                    rules.append(
                        DiscountRule(
                            rule_id=rule_id,
                            description=f"Bulk {category} discount",
                            natural_language=self._generate_nl_description(
                                conditions, "percentage", discount_pct
                            ),
                            conditions=conditions,
                            discount_type="percentage",
                            discount_value=float(discount_pct),
                        )
                    )

        # 6. Category spend threshold rules
        if config["rule_types"]["category_spend"]["enabled"]:
            for category, category_config in config["categories"].items():
                min_spend = (
                    category_config["price_range"][1]
                    * config["rule_types"]["category_spend"]["spend_multiplier"]
                )
                discount_pct = config["rule_types"]["category_spend"]["discount_rate"]
                conditions = {"category": category, "category_min_spend": min_spend}
                rule_id = f"{category}_spend_{int(min_spend)}"

                rules.append(
                    DiscountRule(
                        rule_id=rule_id,
                        description=f"{category.title()} category spend discount",
                        natural_language=self._generate_nl_description(
                            conditions, "percentage", discount_pct
                        ),
                        conditions=conditions,
                        discount_type="percentage",
                        discount_value=float(discount_pct),
                    )
                )

        # 7. Membership tier rules
        for tier, tier_config in config["membership_tiers"].items():
            min_years = tier_config["min_years"]
            discount_boost = tier_config["discount_boost"]
            conditions = {"membership_years": min_years}
            rule_id = f"member_{tier}_{min_years}y"

            rules.append(
                DiscountRule(
                    rule_id=rule_id,
                    description=f"{tier.title()} member discount",
                    natural_language=self._generate_nl_description(
                        conditions, "percentage", discount_boost
                    ),
                    conditions=conditions,
                    discount_type="percentage",
                    discount_value=float(discount_boost),
                )
            )

        return rules

    def get_rule_mappings(self) -> Dict[str, str]:
        """Get a mapping of rule IDs to their natural language descriptions."""
        return {rule.rule_id: rule.natural_language for rule in self.rules}

    def get_rule_count_summary(self) -> Dict[str, int]:
        """Get a summary of how many rules are generated by category."""
        summary = {
            "customer_spend": 0,
            "customer_category": 0,
            "promo_codes": 0,
            "spend_based": 0,
            "bulk_category": 0,
            "category_spend": 0,
            "total": len(self.rules),
        }

        for rule in self.rules:
            rule_id = rule.rule_id
            if any(
                rule_id.startswith(f"{ct}_") and rule_id.split("_")[1].isdigit()
                for ct in self.rule_config["customer_types"].keys()
            ):
                summary["customer_spend"] += 1
            elif any(
                rule_id.startswith(f"{ct}_") and not rule_id.split("_")[1].isdigit()
                for ct in self.rule_config["customer_types"].keys()
            ):
                summary["customer_category"] += 1
            elif rule_id in [
                pc.lower() for pc in self.rule_config["promo_codes"].keys()
            ]:
                summary["promo_codes"] += 1
            elif rule_id.startswith("spend_"):
                summary["spend_based"] += 1
            elif rule_id.startswith("bulk_"):
                summary["bulk_category"] += 1
            elif "_spend_" in rule_id:
                summary["category_spend"] += 1

        return summary

    def generate_random_cart(self) -> Tuple[List[CartItem], float]:
        """Generate a random shopping cart."""
        num_items = random.randint(1, 8)
        cart = []

        item_names = {
            ItemCategory.ELECTRONICS: [
                "Laptop",
                "Phone",
                "Headphones",
                "Tablet",
                "Camera",
            ],
            ItemCategory.CLOTHING: ["T-Shirt", "Jeans", "Dress", "Jacket", "Shoes"],
            ItemCategory.BOOKS: [
                "Novel",
                "Textbook",
                "Cookbook",
                "Biography",
                "Manual",
            ],
            ItemCategory.FOOD: [
                "Snacks",
                "Beverages",
                "Frozen Meal",
                "Fruits",
                "Bread",
            ],
            ItemCategory.HOME: ["Pillow", "Lamp", "Towel", "Plant", "Candle"],
            ItemCategory.SPORTS: [
                "Ball",
                "Weights",
                "Yoga Mat",
                "Running Shoes",
                "Water Bottle",
            ],
            ItemCategory.BEAUTY: [
                "Lipstick",
                "Shampoo",
                "Moisturizer",
                "Perfume",
                "Nail Polish",
            ],
            ItemCategory.HEALTH: [
                "Vitamins",
                "First Aid Kit",
                "Thermometer",
                "Bandages",
                "Hand Sanitizer",
            ],
        }

        price_ranges = {
            ItemCategory.ELECTRONICS: (50, 200),
            ItemCategory.CLOTHING: (10, 50),
            ItemCategory.BOOKS: (10, 30),
            ItemCategory.FOOD: (5, 20),
            ItemCategory.HOME: (10, 40),
            ItemCategory.SPORTS: (20, 60),
            ItemCategory.BEAUTY: (10, 35),
            ItemCategory.HEALTH: (10, 50),
        }

        for _ in range(num_items):
            category = random.choice(list(ItemCategory))
            name = random.choice(item_names[category])
            min_price, max_price = price_ranges[category]
            # Use round numbers that are multiples of 5 or 10
            price_options = [i for i in range(min_price, max_price + 1, 5)]
            price = float(random.choice(price_options))
            quantity = random.randint(1, 3)

            cart.append(CartItem(name, category, price, quantity))

        total_spend = sum(item.price * item.quantity for item in cart)
        return cart, total_spend

    def generate_random_customer(self) -> CustomerProfile:
        """Generate a random customer profile."""
        customer_type = random.choice(list(CustomerType))
        membership_years = random.randint(0, 10)
        return CustomerProfile(customer_type, membership_years)

    def check_rule_applies(
        self,
        rule: DiscountRule,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
    ) -> bool:
        """Check if a discount rule applies to the current purchase."""
        conditions = rule.conditions

        # Check customer type
        if "customer_type" in conditions:
            if customer.customer_type.value != conditions["customer_type"]:
                return False

        # Check minimum spend
        if "min_spend" in conditions:
            if total_spend < conditions["min_spend"]:
                return False

        # Check promo code
        if "promo_code" in conditions:
            if promo_code != conditions["promo_code"]:
                return False

        # Check membership years
        if "membership_years" in conditions:
            if customer.membership_years < conditions["membership_years"]:
                return False

        # Check category-specific conditions
        if "category" in conditions:
            target_category = conditions["category"]
            category_items = [
                item for item in cart if item.category.value == target_category
            ]

            if not category_items:
                return False

            # Check minimum quantity for category
            if "min_quantity" in conditions:
                total_quantity = sum(item.quantity for item in category_items)
                if total_quantity < conditions["min_quantity"]:
                    return False

            # Check minimum spend for category
            if "category_min_spend" in conditions:
                category_spend = sum(
                    item.price * item.quantity for item in category_items
                )
                if category_spend < conditions["category_min_spend"]:
                    return False

        return True

    def calculate_applicable_discounts(
        self,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
    ) -> List[DiscountRule]:
        """Calculate all applicable discount rules for a purchase."""
        applicable_rules = []

        for rule in self.rules:
            if self.check_rule_applies(rule, cart, customer, total_spend, promo_code):
                applicable_rules.append(rule)

        return applicable_rules

    def calculate_final_price(
        self,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
        return_applied_rules: bool = False,
    ):
        """Calculate the final price after applying all applicable discounts with deterministic precedence."""
        applicable_rules = self.calculate_applicable_discounts(
            cart, customer, total_spend, promo_code
        )

        final_price = total_spend
        actually_applied = []

        # Separate rules by type for deterministic application
        category_rules = [
            r
            for r in applicable_rules
            if r.discount_type == "percentage"
            and "category" in r.conditions
            and "min_quantity" not in r.conditions
        ]
        total_percentage_rules = [
            r
            for r in applicable_rules
            if r.discount_type == "percentage" and "category" not in r.conditions
        ]
        fixed_amount_rules = [
            r for r in applicable_rules if r.discount_type == "fixed_amount"
        ]

        # Apply discounts in deterministic order:
        # 1. Category-specific percentage discounts (applied to category totals)
        category_discounts = {}
        for rule in category_rules:
            category = rule.conditions["category"]
            if (
                category not in category_discounts
                or rule.discount_value > category_discounts[category]
            ):
                category_discounts[category] = rule.discount_value

        # Track winning category rules and apply discounts
        for rule in category_rules:
            if rule.discount_value == category_discounts[rule.conditions["category"]]:
                actually_applied.append(rule)

        for category, discount_pct in category_discounts.items():
            category_items = [item for item in cart if item.category.value == category]
            category_total = sum(item.price * item.quantity for item in category_items)
            final_price -= category_total * (discount_pct / 100)

        # 2. Total purchase percentage discount (re-evaluate applicability after category discounts)
        applicable_total_rules = [
            r
            for r in total_percentage_rules
            if self.check_rule_applies(r, cart, customer, final_price, promo_code)
        ]

        if applicable_total_rules:
            winner = max(applicable_total_rules, key=lambda r: r.discount_value)
            actually_applied.append(winner)
            final_price -= final_price * (winner.discount_value / 100)

        # 3. Fixed amount discounts (sum all applicable ones)
        actually_applied.extend(fixed_amount_rules)
        for rule in fixed_amount_rules:
            final_price -= rule.discount_value

        # Ensure price doesn't go below 0
        final_price = max(0, final_price)

        if return_applied_rules:
            return round(final_price, 2), actually_applied
        return round(final_price, 2)

    def get_actually_applied_rules(
        self,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
    ) -> List[DiscountRule]:
        """Get only the rules that are actually applied during discount calculation."""
        _, actually_applied = self.calculate_final_price(
            cart, customer, total_spend, promo_code, return_applied_rules=True
        )
        return actually_applied

    def format_cart_description(
        self,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
    ) -> str:
        """Format cart and customer info for the prompt."""
        cart_items = []
        for item in cart:
            cart_items.append(
                f"- {item.name} ({item.category.value}): ${item.price:.2f} x {item.quantity}"
            )

        cart_str = "\n".join(cart_items)

        promo_str = (
            f"\nPromo code: {promo_code}" if promo_code else "\nPromo code: None"
        )

        description = f"""Customer Profile:
- Type: {customer.customer_type.value}
- Membership years: {customer.membership_years}

Shopping Cart:
{cart_str}

{promo_str}"""

        return description

    def create_prompt(
        self,
        cart: List[CartItem],
        customer: CustomerProfile,
        total_spend: float,
        promo_code: str = None,
        rules_to_include: List[DiscountRule] = None,
        include_rules: bool = True,
    ) -> str:
        """Create a clean prompt with rules at the end if provided."""
        cart_description = self.format_cart_description(
            cart, customer, total_spend, promo_code
        )

        base_prompt = f"""Calculate the final price for the following customer purchase after applying all applicable discount rules.

{cart_description}

IMPORTANT: Apply discounts in this exact order to the running total:
1. Category-specific percentage discounts (apply only the highest discount per category to each category's subtotal)
2. Total purchase percentage discounts (apply only the highest total discount to the remaining amount after step 1)
3. Fixed amount discounts (subtract from the remaining amount after step 2, sum all applicable fixed discounts)

Note: Each discount applies to the current running total, not the original price.

Calculate the final price after applying all applicable discount rules. End your answer with \\boxed{{final price}}."""

        if not include_rules or not rules_to_include:
            return base_prompt

        # Format rules for the prompt
        rules_text = "\n".join(
            [f"- {rule.natural_language}" for rule in rules_to_include]
        )

        prompt_with_rules = f"""Calculate the final price for the following customer purchase after applying all applicable discount rules.

{cart_description}

IMPORTANT: Apply discounts in this exact order to the running total:
1. Category-specific percentage discounts (apply only the highest discount per category to each category's subtotal)
2. Total purchase percentage discounts (apply only the highest total discount to the remaining amount after step 1)
3. Fixed amount discounts (subtract from the remaining amount after step 2, sum all applicable fixed discounts)

Note: Each discount applies to the current running total, not the original price.

Discount Rules:
{rules_text}

Calculate the final price after applying all applicable discount rules. End your answer with \\boxed{{final price}}."""

        return prompt_with_rules


def generate_retail_dataset_variant(
    n: int, rule_engine: RetailRuleEngine, variant: str
) -> List[Dict]:
    """Generate retail discount dataset with specified rule variant.

    Args:
        n: Number of examples to generate
        rule_engine: RetailRuleEngine instance
        variant: One of 'all_rules', 'only_applicable_rules', 'no_rules'
    """
    examples = []

    for _ in range(n):
        # Generate random purchase scenario
        cart, total_spend = rule_engine.generate_random_cart()
        customer = rule_engine.generate_random_customer()

        # Randomly decide if there's a promo code
        promo_codes = [
            "SAVE20",
            "WELCOME10",
            "NEWBIE5",
            "STUDENT15",
            None,
            None,
            None,
            None,
            None,
            None,
        ]
        promo_code = random.choice(promo_codes)

        # Calculate applicable discounts for metadata
        applicable_rule_objects = rule_engine.calculate_applicable_discounts(
            cart, customer, total_spend, promo_code
        )

        # Create prompt based on variant
        if variant == "all_rules":
            prompt_content = rule_engine.create_prompt(
                cart,
                customer,
                total_spend,
                promo_code,
                rules_to_include=rule_engine.rules,
                include_rules=True,
            )
        elif variant == "only_applicable_rules":
            # Get rules that are actually applied after all discount calculations
            actually_applied_rules = rule_engine.get_actually_applied_rules(
                cart, customer, total_spend, promo_code
            )
            prompt_content = rule_engine.create_prompt(
                cart,
                customer,
                total_spend,
                promo_code,
                rules_to_include=actually_applied_rules,
                include_rules=True,
            )
        elif variant == "no_rules":
            prompt_content = rule_engine.create_prompt(
                cart,
                customer,
                total_spend,
                promo_code,
                rules_to_include=None,
                include_rules=False,
            )
        else:
            raise ValueError(
                f"Invalid variant: {variant}. Must be one of 'all_rules', 'only_applicable_rules', 'no_rules'"
            )

        # Calculate the final price as ground truth
        final_price = rule_engine.calculate_final_price(
            cart, customer, total_spend, promo_code
        )

        examples.append(
            {
                "prompt": [{"role": "user", "content": prompt_content}],
                "reward_model": {
                    "ground_truth": str(final_price),
                    "applicable_rules": [
                        rule.rule_id for rule in applicable_rule_objects
                    ],
                    "total_spend": total_spend,
                    "final_price": final_price,
                    "customer_type": customer.customer_type.value,
                    "promo_code": promo_code,
                    "variant": variant,
                },
            }
        )

    return examples


def main():
    # Configuration
    eval_size = 256

    # Initialize rule engine
    rule_engine = RetailRuleEngine()

    print(f"Generated {len(rule_engine.rules)} discount rules:")
    for rule in rule_engine.rules:
        print(f"- {rule.rule_id}: {rule.natural_language}")

    # Generate all three variants for training and evaluation
    variants = ["all_rules", "only_applicable_rules", "no_rules"]

    for variant in variants:
        print(f"Generating {eval_size} evaluation examples for variant: {variant}")
        eval_examples = generate_retail_dataset_variant(eval_size, rule_engine, variant)

        # Save datasets with variant suffix
        eval_filename = f"data/retail_discount_eval_{eval_size}_{variant}.parquet"

        pd.DataFrame(eval_examples).to_parquet(eval_filename)

        print(f"Saved {len(eval_examples)} evaluation examples to {eval_filename}")

    # Load and preview examples from all_rules variant
    print("\n" + "=" * 80)
    print("SAMPLE DATASET PREVIEW (all_rules variant)")
    print("=" * 80)

    sample_dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"data/retail_discount_eval_{eval_size}_all_rules.parquet"
        },
    )["train"]
    print("Sample all rules evaluation example:")
    print(json.dumps(sample_dataset[0], indent=2))

    sample_applicable_dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"data/retail_discount_eval_{eval_size}_only_applicable_rules.parquet"
        },
    )["train"]
    print("Sample applicable rules evaluation example:")
    print(json.dumps(sample_applicable_dataset[0], indent=2))

    sample_no_rules_dataset = load_dataset(
        "parquet",
        data_files={"train": f"data/retail_discount_eval_{eval_size}_no_rules.parquet"},
    )["train"]
    print("Sample no rules evaluation example:")
    print(json.dumps(sample_no_rules_dataset[0], indent=2))


if __name__ == "__main__":
    main()
