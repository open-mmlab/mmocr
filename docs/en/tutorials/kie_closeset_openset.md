# KIE: Difference between CloseSet & OpenSet

Being trained on WildReceipt, SDMG-R, or other KIE models, can identify the types of text boxes on a receipt picture.
But what SDMG-R can do is far more beyond that. For example, it's able to identify key-value pairs on the picture. To demonstrate such ability and hopefully facilitate future research, we release a demonstrative version of WildReceiptOpenset annotated in OpenSet format, and provide a full training/testing pipeline for KIE models such as SDMG-R.
Since it might be a *confusing* update, we'll elaborate on the key differences between the OpenSet and CloseSet format, taking WildReceipt as an example.

## CloseSet

WildReceipt ("CloseSet") divides text boxes into 26 categories. There are 12 key-value pairs of fine-grained key information categories, such as (`Prod_item_value`, `Prod_item_key`), (`Prod_price_value`, `Prod_price_key`) and (`Tax_value`, `Tax_key`), plus two more "do not care" categories: `Ignore` and `Others`.

The objective of CloseSet SDMGR is to predict which category fits the text box best, but it will not predict the relations among text boxes. For instance, if there are four text boxes "Hamburger", "Hotdog", "$1" and "$2" on the receipt, the model may assign `Prod_item_value` to the first two boxes and `Prod_price_value` to the last two, but it can't tell if Hamburger sells for $1 or $2. However, this could be achieved in the open-set variant.

<div align="center">
    <img src="https://raw.githubusercontent.com/open-mmlab/mmocr/main/demo/resources/demo_kie_pred.png"/><br>
</div>
<br>

```{warning}

A `*_key` and `*_value` pair do not necessarily have to both appear on the receipt. For example, we usually won't see `Prod_item_key` appearing on the receipt, while there can be multiple boxes annotated as `Pred_item_value`. In contrast, `Tax_key` and `Tax_value` are likely to appear together since they're usually structured as `Tax`: `11.02` on the receipt.

```

## OpenSet

In OpenSet, all text boxes, or nodes, have only 4 possible categories: `background`, `key`, `value`, and `others`. The connectivity between nodes are annotated as *edge labels*. If a pair of key-value nodes have the same edge label, they are connected by an valid edge.

Multiple nodes can have the same edge label. However, only key and value nodes will be linked by edges. The nodes of same category will never be connected.

When making OpenSet annotations, each node must have an edge label. It should be an unique one if it falls into non-`key` non-`value` categories.

```{note}
You can merge `background` to `others` if telling background apart is not important, and we provide this choice in the conversion script for WildReceipt .
```

### Converting WildReceipt from CloseSet to OpenSet

We provide a [conversion script](../datasets/kie.md) that converts WildRecipt-like dataset to OpenSet format. This script links every `key`-`value` pairs following the rules above. Here's an example illustration: (For better understanding, all the node labels are presented as texts)

| box_content | closeset_node_label | closeset_edge_label | openset_node_label | openset_edge_label |
| :---------: | :-----------------: | :-----------------: | :----------------: | :----------------: |
|    hello    |       Ignore        |          -          |       Others       |         0          |
|    world    |       Ignore        |          -          |       Others       |         1          |
|    Actor    |      Actor_key      |          -          |        Key         |         2          |
|     Tom     |     Actor_value     |          -          |       Value        |         2          |
|    Tony     |     Actor_value     |          -          |       Value        |         2          |
|     Tim     |     Actor_value     |          -          |       Value        |         2          |
|  something  |       Ignore        |          -          |       Others       |         3          |
|   Actress   |     Actress_key     |          -          |        Key         |         4          |
|    Lucy     |    Actress_value    |          -          |       Value        |         4          |
|    Zora     |    Actress_value    |          -          |       Value        |         4          |

```{warning}

A common request from our community is to extract the relations between food items and food prices. In this case, this conversion script ***is not you need***.
Wildrecipt doesn't provide necessary information to recover this relation. For instance, there are four text boxes "Hamburger", "Hotdog", "$1" and "$2" on the receipt, and here's how they actually look like before and after the conversion:

|box_content | closeset_node_label| closeset_edge_label | openset_node_label | openset_edge_label |
| :----: | :---: | :----: | :---: | :---: |
| Hamburger | Prod_item_value | - | Value | 0 |
| Hotdog | Prod_item_value | - | Value | 0 |
| $1 | Prod_price_value | - | Value | 1 |
| $2 | Prod_price_value  | - | Value | 1 |

So there won't be any valid edges connecting them. Nevertheless, OpenSet format is far more general than CloseSet, so this task can be achieved by annotating the data from scratch.

|box_content | openset_node_label | openset_edge_label |
| :----: | :---: | :---: |
| Hamburger | Value | 0 |
| Hotdog | Value | 1 |
| $1 | Value | 0 |
| $2 | Value | 1 |

```
