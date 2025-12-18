#!/usr/bin/env python3
"""
Configuration and Customization System Demo

This script demonstrates the enhanced configuration and customization capabilities
of the VectorBT Visualization Enhancement system.
"""

import sys
import os
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_predictor.visualization import (
    PlotConfig, PortfolioConfig, ConfigurationManager, ConfigurationValidator
)


def demo_preset_configurations():
    """Demonstrate preset configurations for different use cases."""
    print("=== Preset Configurations Demo ===")
    
    # Create different presets
    presets = ['trading', 'research', 'presentation', 'dashboard']
    
    for preset_name in presets:
        config = PlotConfig.create_preset(preset_name)
        print(f"\n{preset_name.title()} Preset:")
        print(f"  Dimensions: {config.width}x{config.height}")
        print(f"  Theme: {config.theme}")
        print(f"  Show trades: {config.show_trades}")
        print(f"  Show metrics: {config.show_metrics}")
        print(f"  Export formats: {config.export_formats}")


def demo_theme_application():
    """Demonstrate theme application and customization."""
    print("\n=== Theme Application Demo ===")
    
    # Start with a base configuration
    base_config = PlotConfig.create_preset('trading')
    
    # Apply different themes
    themes = ['default', 'dark', 'professional', 'minimal', 'colorful']
    
    for theme_name in themes:
        themed_config = base_config.apply_theme(theme_name)
        print(f"\n{theme_name.title()} Theme:")
        print(f"  Template: {themed_config.template}")
        print(f"  Background: {themed_config.background_color}")
        print(f"  Primary color: {themed_config.primary_color}")
        print(f"  Text color: {themed_config.text_color}")


def demo_custom_colors():
    """Demonstrate custom color palette application."""
    print("\n=== Custom Color Palette Demo ===")
    
    base_config = PlotConfig.create_preset('trading')
    
    # Define custom color palettes
    palettes = {
        'ocean': {
            'primary': '#006994',
            'secondary': '#47B5FF',
            'success': '#06FFA5',
            'danger': '#FF6B6B',
            'background': '#F0F8FF'
        },
        'sunset': {
            'primary': '#FF6B35',
            'secondary': '#F7931E',
            'success': '#FFD23F',
            'danger': '#EE4B2B',
            'background': '#FFF8DC'
        }
    }
    
    for palette_name, colors in palettes.items():
        custom_config = base_config.customize_colors(colors)
        print(f"\n{palette_name.title()} Palette:")
        for color_name, color_value in colors.items():
            print(f"  {color_name}: {color_value}")


def demo_user_preferences():
    """Demonstrate user preference management."""
    print("\n=== User Preferences Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir)
        
        # Create and save different user preferences
        preferences = {
            'day_trading': {
                'plot': PlotConfig.create_preset('trading').apply_theme('dark'),
                'portfolio': PortfolioConfig(init_cash=10000, fees=0.005, size_strategy='percent_equity', size_value=0.1)
            },
            'swing_trading': {
                'plot': PlotConfig.create_preset('research').apply_theme('professional'),
                'portfolio': PortfolioConfig(init_cash=50000, fees=0.002, size_strategy='fixed_amount', size_value=2000)
            },
            'long_term': {
                'plot': PlotConfig.create_preset('presentation').apply_theme('minimal'),
                'portfolio': PortfolioConfig(init_cash=100000, fees=0.001, size_strategy='percent_equity', size_value=0.05)
            }
        }
        
        # Save preferences
        for pref_name, configs in preferences.items():
            config_manager.save_user_preferences(
                configs['plot'], 
                configs['portfolio'], 
                pref_name
            )
            print(f"Saved preference: {pref_name}")
        
        # List all preferences
        pref_list = config_manager.list_user_preferences()
        print(f"\nTotal preferences saved: {len(pref_list)}")
        
        # Load and display a preference
        loaded_pref = config_manager.load_user_preferences('day_trading')
        plot_config = loaded_pref['plot_config']
        portfolio_config = loaded_pref['portfolio_config']
        
        print(f"\nLoaded 'day_trading' preference:")
        print(f"  Plot theme: {plot_config.theme}")
        print(f"  Portfolio capital: ${portfolio_config.init_cash:,}")
        print(f"  Position sizing: {portfolio_config.size_strategy}")


def demo_configuration_templates():
    """Demonstrate configuration template management."""
    print("\n=== Configuration Templates Demo ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir)
        
        # Create custom templates
        templates = {
            'crypto_trading': {
                'plot': PlotConfig(
                    width=1400, height=800, theme='dark',
                    show_trades=True, show_volatility=True,
                    marker_size=12, enable_animations=True
                ),
                'description': 'High-volatility crypto trading analysis',
                'analysis_type': 'crypto'
            },
            'forex_analysis': {
                'plot': PlotConfig(
                    width=1200, height=600, theme='professional',
                    show_benchmark=True, benchmark_symbol='DXY',
                    show_sharpe_ratio=True, show_var=True
                ),
                'description': 'Forex pair analysis with currency index benchmark',
                'analysis_type': 'forex'
            }
        }
        
        # Save templates
        for template_name, template_data in templates.items():
            config_manager.create_configuration_template(
                template_name,
                template_data['plot'],
                description=template_data['description'],
                analysis_type=template_data['analysis_type']
            )
            print(f"Created template: {template_name}")
        
        # List templates by analysis type
        crypto_templates = config_manager.list_configuration_templates('crypto')
        forex_templates = config_manager.list_configuration_templates('forex')
        
        print(f"\nCrypto templates: {len(crypto_templates)}")
        print(f"Forex templates: {len(forex_templates)}")
        
        # Load and display a template
        loaded_template = config_manager.load_configuration_template('crypto_trading')
        print(f"\nLoaded 'crypto_trading' template:")
        print(f"  Description: {loaded_template['description']}")
        print(f"  Analysis type: {loaded_template['analysis_type']}")
        print(f"  Animations enabled: {loaded_template['plot_config'].enable_animations}")


def demo_runtime_updates():
    """Demonstrate runtime configuration updates."""
    print("\n=== Runtime Configuration Updates Demo ===")
    
    config_manager = ConfigurationManager()
    
    # Start with base configuration
    base_config = PlotConfig.create_preset('trading')
    print(f"Original dimensions: {base_config.width}x{base_config.height}")
    print(f"Original theme: {base_config.theme}")
    
    # Apply runtime updates
    updates = {
        'width': 1600,
        'height': 900,
        'theme': 'dark',
        'show_annotations': False,
        'marker_size': 10,
        'export_dpi': 600
    }
    
    updated_config = config_manager.update_configuration_runtime(base_config, updates)
    
    print(f"\nAfter updates:")
    print(f"  Dimensions: {updated_config.width}x{updated_config.height}")
    print(f"  Theme: {updated_config.theme}")
    print(f"  Annotations: {updated_config.show_annotations}")
    print(f"  Marker size: {updated_config.marker_size}")
    print(f"  Export DPI: {updated_config.export_dpi}")


def demo_configuration_validation():
    """Demonstrate configuration validation and suggestions."""
    print("\n=== Configuration Validation Demo ===")
    
    validator = ConfigurationValidator()
    
    # Create configurations with potential issues
    plot_config = PlotConfig(
        width=400,  # Too small
        height=300,  # Too small
        marker_size=2,  # Too small
        show_metrics=False,  # Missing important info
        export_dpi=600  # Very high
    )
    
    portfolio_config = PortfolioConfig(
        init_cash=500,  # Very low
        fees=0.02,  # Very high (2%)
        size_value=200,  # Large relative to capital
        stop_loss=None  # No risk management
    )
    
    # Validate compatibility
    validation_result = validator.validate_configuration_compatibility(plot_config, portfolio_config)
    
    print("Validation Results:")
    if validation_result['errors']:
        print("  Errors:")
        for error in validation_result['errors']:
            print(f"    - {error}")
    
    if validation_result['warnings']:
        print("  Warnings:")
        for warning in validation_result['warnings']:
            print(f"    - {warning}")
    
    # Get improvement suggestions
    plot_suggestions = validator.suggest_configuration_improvements(plot_config)
    portfolio_suggestions = validator.suggest_configuration_improvements(portfolio_config)
    
    print("\nPlot Configuration Suggestions:")
    for suggestion in plot_suggestions:
        print(f"  - {suggestion}")
    
    print("\nPortfolio Configuration Suggestions:")
    for suggestion in portfolio_suggestions:
        print(f"  - {suggestion}")


def main():
    """Run all configuration system demonstrations."""
    print("VectorBT Visualization Configuration System Demo")
    print("=" * 50)
    
    try:
        demo_preset_configurations()
        demo_theme_application()
        demo_custom_colors()
        demo_user_preferences()
        demo_configuration_templates()
        demo_runtime_updates()
        demo_configuration_validation()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()