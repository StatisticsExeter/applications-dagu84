import pandas as pd
import pytest
from plotly.graph_objs import Figure
from course.exercises.exercises_3 import (
    box_plot, scatterplot, scatterplot_groups,
    scatterplot_matrix, bar_chart_means, stacked_bar_counts
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie"],
        "island": ["Torgersen", "Dream", "Biscoe", "Dream"],
        "bill_length_mm": [39.1, 48.7, 50.0, 38.9],
        "bill_depth_mm": [18.7, 17.4, 15.3, 17.8],
        "body_mass_g": [3750, 3800, 5000, 3700]
    })


def test_box_plot(sample_df):
    fig = box_plot(sample_df, "species", "bill_length_mm")
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "box"


def test_scatterplot(sample_df):
    fig = scatterplot(sample_df, "bill_length_mm", "body_mass_g")
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "scatter"


def test_scatterplot_groups(sample_df):
    fig = scatterplot_groups(sample_df, "bill_length_mm", "body_mass_g", "species")
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "scatter"


def test_scatterplot_matrix(sample_df):
    fig = scatterplot_matrix(sample_df, ["bill_length_mm", "bill_depth_mm", "body_mass_g"])
    assert isinstance(fig, Figure)
    assert len(fig.data) > 0


def test_bar_chart_means(sample_df):
    labels = {"species": "Penguin Species", "body_mass_g": "Mass (g)"}
    fig = bar_chart_means(sample_df, "species", "body_mass_g", labels)
    assert isinstance(fig, Figure)
    assert fig.layout.title.text.startswith("Average body_mass_g")


def test_stacked_bar_counts(sample_df):
    labels = {"species": "Species", "island": "Island"}
    fig = stacked_bar_counts(sample_df, "species", "island", labels)
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "bar"
