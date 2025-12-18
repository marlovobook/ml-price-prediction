"""
Plot Export Engine for VectorBT Visualization Enhancement.

This module handles exporting VectorBT visualizations in multiple formats
and organizing output files for reports and presentations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime

from .portfolio_config import PlotConfig, VisualizationResult
from ..utils.exceptions import DataValidationError


class PlotExportEngine:
    """
    Engine for exporting VectorBT plots and data in multiple formats.
    
    Supports exporting to PNG, HTML, SVG, PDF, and CSV formats with
    organized directory structure and metadata preservation.
    """
    
    def __init__(self, export_config: Optional[PlotConfig] = None):
        """
        Initialize the Plot Export Engine.
        
        Args:
            export_config: Configuration for export options
        """
        self.config = export_config or PlotConfig()
        self.logger = logging.getLogger(__name__)
        
        # Create export directory if it doesn't exist
        self.export_dir = Path(self.config.export_directory)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Plot Export Engine initialized with directory: {self.export_dir}")
    
    def export_plot(
        self,
        plot_object: Any,
        filename: str,
        formats: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export plot in specified formats.
        
        Args:
            plot_object: VectorBT plot object or Plotly figure
            filename: Base filename for export (without extension)
            formats: List of export formats (defaults to config)
            metadata: Optional metadata to include with exports
            
        Returns:
            Dictionary mapping format to file path
            
        Raises:
            DataValidationError: If export fails
        """
        try:
            start_time = time.time()
            formats = formats or self.config.export_formats
            
            # Validate formats
            valid_formats = {'png', 'html', 'svg', 'pdf', 'json'}
            invalid_formats = set(formats) - valid_formats
            if invalid_formats:
                raise DataValidationError(f"Invalid export formats: {invalid_formats}")
            
            export_paths = {}
            
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = self.export_dir / f"{filename}_{timestamp}"
            export_subdir.mkdir(exist_ok=True)
            
            self.logger.info(f"Exporting plot '{filename}' in formats: {formats}")
            
            # Export in each requested format
            for fmt in formats:
                try:
                    file_path = export_subdir / f"{filename}.{fmt}"
                    
                    if fmt == 'png':
                        self._export_png(plot_object, file_path)
                    elif fmt == 'html':
                        self._export_html(plot_object, file_path)
                    elif fmt == 'svg':
                        self._export_svg(plot_object, file_path)
                    elif fmt == 'pdf':
                        self._export_pdf(plot_object, file_path)
                    elif fmt == 'json':
                        self._export_json(plot_object, file_path)
                    
                    export_paths[fmt] = str(file_path)
                    self.logger.debug(f"Exported {fmt.upper()}: {file_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to export {fmt}: {str(e)}")
                    export_paths[f"{fmt}_error"] = str(e)
            
            # Export metadata if provided
            if metadata:
                metadata_path = export_subdir / f"{filename}_metadata.json"
                self._export_metadata(metadata, metadata_path)
                export_paths['metadata'] = str(metadata_path)
            
            export_time = time.time() - start_time
            self.logger.info(
                f"Plot export completed in {export_time:.2f}s. "
                f"Exported {len([k for k in export_paths.keys() if not k.endswith('_error')])} formats"
            )
            
            return export_paths
            
        except Exception as e:
            self.logger.error(f"Error exporting plot: {str(e)}")
            raise DataValidationError(f"Plot export failed: {str(e)}")
    
    def export_plot_data(
        self,
        data: Dict[str, pd.Series],
        filename: str,
        include_summary: bool = True
    ) -> str:
        """
        Export underlying plot data as CSV.
        
        Args:
            data: Dictionary of data series to export
            filename: Base filename for export
            include_summary: Whether to include summary statistics
            
        Returns:
            Path to exported CSV file
        """
        try:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = self.export_dir / f"{filename}_data_{timestamp}.csv"
            
            # Combine all series into DataFrame
            combined_df = pd.DataFrame(data)
            
            # Add summary statistics if requested
            if include_summary:
                summary_df = combined_df.describe()
                # Add empty row separator
                separator = pd.DataFrame([['---'] * len(combined_df.columns)], 
                                       columns=combined_df.columns)
                combined_df = pd.concat([combined_df, separator, summary_df])
            
            # Export to CSV
            combined_df.to_csv(csv_path, index=True)
            
            self.logger.info(f"Plot data exported to: {csv_path}")
            return str(csv_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting plot data: {str(e)}")
            raise DataValidationError(f"Data export failed: {str(e)}")
    
    def export_visualization_result(
        self,
        result: VisualizationResult,
        filename: str,
        export_data: bool = True
    ) -> Dict[str, str]:
        """
        Export complete visualization result including plot and data.
        
        Args:
            result: VisualizationResult to export
            filename: Base filename for exports
            export_data: Whether to export underlying data
            
        Returns:
            Dictionary of all exported file paths
        """
        try:
            if not result.success:
                raise DataValidationError(f"Cannot export failed result: {result.error_message}")
            
            export_paths = {}
            
            # Export plot in configured formats
            if result.plot_object is not None:
                plot_paths = self.export_plot(
                    result.plot_object,
                    filename,
                    metadata={
                        'generation_time': result.generation_time,
                        'metrics': result.metrics_summary,
                        'success': result.success
                    }
                )
                export_paths.update(plot_paths)
            
            # Export underlying data if requested
            if export_data and result.plot_data:
                data_path = self.export_plot_data(result.plot_data, filename)
                export_paths['data_csv'] = data_path
            
            # Export metrics summary
            if result.metrics_summary:
                metrics_path = self._export_metrics_summary(result.metrics_summary, filename)
                export_paths['metrics_json'] = metrics_path
            
            return export_paths
            
        except Exception as e:
            self.logger.error(f"Error exporting visualization result: {str(e)}")
            raise DataValidationError(f"Result export failed: {str(e)}")
    
    def create_comprehensive_report(
        self,
        results: Dict[str, VisualizationResult],
        report_title: str = "VectorBT Analysis Report",
        template_type: str = "standard",
        include_raw_data: bool = False,
        custom_sections: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create comprehensive PDF/HTML report with embedded visualizations.
        
        This method generates professional-grade reports with:
        - Executive summary with key metrics
        - Individual strategy analysis sections
        - Comparative performance analysis
        - Risk analysis and drawdown periods
        - Embedded interactive visualizations
        - Raw data exports (optional)
        
        Args:
            results: Dictionary of named visualization results
            report_title: Title for the report
            template_type: Report template ('standard', 'executive', 'technical', 'comparison')
            include_raw_data: Whether to include raw data tables
            custom_sections: Optional custom sections to include
            
        Returns:
            Path to generated report file
            
        Raises:
            DataValidationError: If report generation fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create report directory
            report_dir = self.export_dir / f"report_{timestamp}"
            report_dir.mkdir(exist_ok=True)
            
            # Generate HTML report
            html_path = report_dir / f"{report_title.replace(' ', '_')}_report.html"
            html_content = self._generate_comprehensive_html_report(
                results, report_title, template_type, include_raw_data, custom_sections
            )
            
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Generate PDF version if possible
            pdf_path = self._generate_pdf_report(html_content, report_dir, report_title)
            
            # Export individual visualizations to report directory
            viz_dir = report_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            exported_viz = {}
            for name, result in results.items():
                if result.success and result.plot_object is not None:
                    viz_paths = self.export_plot(
                        result.plot_object,
                        f"{name}_visualization",
                        formats=['png', 'html', 'svg'],
                        metadata={
                            'strategy_name': name,
                            'generation_time': result.generation_time,
                            'metrics': result.metrics_summary
                        }
                    )
                    
                    # Copy files to report viz directory (instead of moving to avoid conflicts)
                    for fmt, path in viz_paths.items():
                        if not fmt.endswith('_error'):
                            src_path = Path(path)
                            if src_path.exists():
                                dst_path = viz_dir / f"{name}_{fmt}.{fmt}"
                                try:
                                    # Use copy instead of rename to avoid conflicts
                                    import shutil
                                    shutil.copy2(src_path, dst_path)
                                    exported_viz[f"{name}_{fmt}"] = str(dst_path)
                                except Exception as copy_error:
                                    self.logger.warning(f"Failed to copy {src_path} to {dst_path}: {copy_error}")
                                    # Keep original path if copy fails
                                    exported_viz[f"{name}_{fmt}"] = str(src_path)
            
            # Export raw data if requested
            if include_raw_data:
                data_dir = report_dir / "data"
                data_dir.mkdir(exist_ok=True)
                
                for name, result in results.items():
                    if result.plot_data:
                        data_path = self.export_plot_data(
                            result.plot_data,
                            f"{name}_data"
                        )
                        # Copy to report data directory
                        src_path = Path(data_path)
                        if src_path.exists():
                            dst_path = data_dir / f"{name}_data.csv"
                            try:
                                import shutil
                                shutil.copy2(src_path, dst_path)
                            except Exception as copy_error:
                                self.logger.warning(f"Failed to copy data file {src_path} to {dst_path}: {copy_error}")
            
            # Create report manifest
            manifest = self._create_report_manifest(
                results, report_title, template_type, html_path, pdf_path, exported_viz
            )
            
            manifest_path = report_dir / "report_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive report created: {html_path}")
            if pdf_path:
                self.logger.info(f"PDF report created: {pdf_path}")
            
            return str(html_path)
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive report: {str(e)}")
            raise DataValidationError(f"Comprehensive report creation failed: {str(e)}")
    
    def create_batch_reports(
        self,
        strategy_results: Dict[str, Dict[str, VisualizationResult]],
        comparison_title: str = "Multi-Strategy Analysis Report"
    ) -> Dict[str, str]:
        """
        Create batch reports for multiple strategies with comparative analysis.
        
        This method generates:
        - Individual strategy reports
        - Comparative analysis report
        - Performance ranking report
        - Risk analysis report
        
        Args:
            strategy_results: Nested dict {strategy_name: {viz_type: result}}
            comparison_title: Title for the comparison report
            
        Returns:
            Dictionary mapping report type to file path
            
        Raises:
            DataValidationError: If batch report generation fails
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_dir = self.export_dir / f"batch_analysis_{timestamp}"
            batch_dir.mkdir(exist_ok=True)
            
            generated_reports = {}
            
            # Generate individual strategy reports
            individual_dir = batch_dir / "individual_strategies"
            individual_dir.mkdir(exist_ok=True)
            
            for strategy_name, results in strategy_results.items():
                try:
                    # Create individual report
                    individual_report = self.create_comprehensive_report(
                        results,
                        f"{strategy_name} Strategy Analysis",
                        template_type="technical",
                        include_raw_data=True
                    )
                    
                    # Copy to batch directory
                    src_path = Path(individual_report).parent
                    dst_path = individual_dir / f"{strategy_name}_report"
                    if src_path.exists():
                        try:
                            import shutil
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                            generated_reports[f"{strategy_name}_individual"] = str(dst_path / f"{strategy_name}_Strategy_Analysis_report.html")
                        except Exception as copy_error:
                            self.logger.warning(f"Failed to copy report directory {src_path} to {dst_path}: {copy_error}")
                            # Keep original path if copy fails
                            generated_reports[f"{strategy_name}_individual"] = individual_report
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create individual report for {strategy_name}: {str(e)}")
            
            # Generate comparative analysis report
            comparison_report = self._generate_comparative_report(
                strategy_results, comparison_title, batch_dir
            )
            generated_reports["comparative_analysis"] = comparison_report
            
            # Generate performance ranking report
            ranking_report = self._generate_ranking_report(
                strategy_results, batch_dir
            )
            generated_reports["performance_ranking"] = ranking_report
            
            # Generate risk analysis report
            risk_report = self._generate_risk_analysis_report(
                strategy_results, batch_dir
            )
            generated_reports["risk_analysis"] = risk_report
            
            # Create batch manifest
            batch_manifest = {
                'batch_title': comparison_title,
                'generation_timestamp': datetime.now().isoformat(),
                'strategies_analyzed': list(strategy_results.keys()),
                'reports_generated': generated_reports,
                'total_strategies': len(strategy_results),
                'batch_directory': str(batch_dir)
            }
            
            manifest_path = batch_dir / "batch_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(batch_manifest, f, indent=2, default=str)
            
            self.logger.info(f"Batch reports created in: {batch_dir}")
            return generated_reports
            
        except Exception as e:
            self.logger.error(f"Error creating batch reports: {str(e)}")
            raise DataValidationError(f"Batch report creation failed: {str(e)}")
    
    def create_report(
        self,
        results: Dict[str, VisualizationResult],
        report_title: str = "VectorBT Analysis Report"
    ) -> str:
        """
        Create comprehensive HTML report with multiple visualizations.
        
        This is a simplified wrapper around create_comprehensive_report for backward compatibility.
        
        Args:
            results: Dictionary of named visualization results
            report_title: Title for the report
            
        Returns:
            Path to generated HTML report
        """
        return self.create_comprehensive_report(
            results, report_title, template_type="standard"
        )
    
    def _export_png(self, plot_object: Any, file_path: Path) -> None:
        """Export plot as PNG image."""
        try:
            # Try VectorBT/Plotly export
            if hasattr(plot_object, 'write_image'):
                plot_object.write_image(
                    str(file_path),
                    width=self.config.width,
                    height=self.config.height,
                    scale=self.config.export_dpi / 72  # Convert DPI to scale
                )
            else:
                raise ValueError("Plot object does not support PNG export")
                
        except Exception as e:
            raise DataValidationError(f"PNG export failed: {str(e)}")
    
    def _export_html(self, plot_object: Any, file_path: Path) -> None:
        """Export plot as interactive HTML."""
        try:
            if hasattr(plot_object, 'write_html'):
                plot_object.write_html(str(file_path))
            elif hasattr(plot_object, 'to_html'):
                with open(file_path, 'w') as f:
                    f.write(plot_object.to_html())
            else:
                raise ValueError("Plot object does not support HTML export")
                
        except Exception as e:
            raise DataValidationError(f"HTML export failed: {str(e)}")
    
    def _export_svg(self, plot_object: Any, file_path: Path) -> None:
        """Export plot as SVG vector image."""
        try:
            if hasattr(plot_object, 'write_image'):
                plot_object.write_image(str(file_path), format='svg')
            else:
                raise ValueError("Plot object does not support SVG export")
                
        except Exception as e:
            raise DataValidationError(f"SVG export failed: {str(e)}")
    
    def _export_pdf(self, plot_object: Any, file_path: Path) -> None:
        """Export plot as PDF."""
        try:
            if hasattr(plot_object, 'write_image'):
                plot_object.write_image(str(file_path), format='pdf')
            else:
                raise ValueError("Plot object does not support PDF export")
                
        except Exception as e:
            raise DataValidationError(f"PDF export failed: {str(e)}")
    
    def _export_json(self, plot_object: Any, file_path: Path) -> None:
        """Export plot configuration as JSON."""
        try:
            if hasattr(plot_object, 'to_json'):
                with open(file_path, 'w') as f:
                    f.write(plot_object.to_json())
            elif hasattr(plot_object, 'to_dict'):
                with open(file_path, 'w') as f:
                    json.dump(plot_object.to_dict(), f, indent=2, default=str)
            else:
                # Fallback: export basic plot info
                plot_info = {
                    'type': str(type(plot_object)),
                    'export_timestamp': datetime.now().isoformat(),
                    'note': 'Limited JSON export - plot object does not support full serialization'
                }
                with open(file_path, 'w') as f:
                    json.dump(plot_info, f, indent=2)
                    
        except Exception as e:
            raise DataValidationError(f"JSON export failed: {str(e)}")
    
    def _export_metadata(self, metadata: Dict[str, Any], file_path: Path) -> None:
        """Export metadata as JSON."""
        try:
            # Add export timestamp
            metadata_with_timestamp = {
                **metadata,
                'export_timestamp': datetime.now().isoformat(),
                'export_engine_version': '1.0.0'
            }
            
            with open(file_path, 'w') as f:
                json.dump(metadata_with_timestamp, f, indent=2, default=str)
                
        except Exception as e:
            raise DataValidationError(f"Metadata export failed: {str(e)}")
    
    def _export_metrics_summary(self, metrics: Dict[str, float], filename: str) -> str:
        """Export metrics summary as JSON."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_path = self.export_dir / f"{filename}_metrics_{timestamp}.json"
            
            metrics_with_metadata = {
                'metrics': metrics,
                'export_timestamp': datetime.now().isoformat(),
                'filename': filename
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics_with_metadata, f, indent=2, default=str)
            
            return str(metrics_path)
            
        except Exception as e:
            raise DataValidationError(f"Metrics export failed: {str(e)}")
    
    def _generate_comprehensive_html_report(
        self,
        results: Dict[str, VisualizationResult],
        title: str,
        template_type: str = "standard",
        include_raw_data: bool = False,
        custom_sections: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate comprehensive HTML report with advanced features.
        
        Args:
            results: Dictionary of visualization results
            title: Report title
            template_type: Type of template to use
            include_raw_data: Whether to include data tables
            custom_sections: Custom sections to add
            
        Returns:
            Complete HTML report content
        """
        # Get template-specific styling
        css_styles = self._get_template_styles(template_type)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(results)
        
        html_parts = [
            f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{title}</title>
                <style>
                    {css_styles}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <div class="report-container">
                    <header class="report-header">
                        <h1>{title}</h1>
                        <div class="report-meta">
                            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p>Template: {template_type.title()}</p>
                            <p>Strategies Analyzed: {len(results)}</p>
                        </div>
                    </header>
            """
        ]
        
        # Add executive summary for executive template
        if template_type == "executive":
            html_parts.append(self._generate_executive_summary(summary_stats))
        
        # Add table of contents
        html_parts.append(self._generate_table_of_contents(results, custom_sections))
        
        # Add summary statistics section
        html_parts.append(self._generate_summary_section(summary_stats))
        
        # Add individual strategy sections
        for name, result in results.items():
            if result.success:
                html_parts.append(
                    self._generate_strategy_section(name, result, template_type, include_raw_data)
                )
        
        # Add comparative analysis if multiple strategies
        if len(results) > 1:
            html_parts.append(self._generate_comparative_section(results))
        
        # Add custom sections if provided
        if custom_sections:
            for section_title, section_content in custom_sections.items():
                html_parts.append(f"""
                <section class="custom-section">
                    <h2>{section_title}</h2>
                    <div class="custom-content">
                        {section_content}
                    </div>
                </section>
                """)
        
        # Add footer
        html_parts.append("""
                    <footer class="report-footer">
                        <p>Generated by VectorBT Visualization Enhancement Engine</p>
                        <p>For more information, visit the project documentation</p>
                    </footer>
                </div>
            </body>
            </html>
        """)
        
        return ''.join(html_parts)
    
    def _generate_html_report(
        self, 
        results: Dict[str, VisualizationResult], 
        title: str
    ) -> str:
        """Generate basic HTML report (backward compatibility)."""
        return self._generate_comprehensive_html_report(results, title, "standard")
    
    def _get_template_styles(self, template_type: str) -> str:
        """Get CSS styles for different report templates."""
        base_styles = """
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background-color: #f8f9fa;
            }
            .report-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            .report-header {
                text-align: center;
                margin-bottom: 40px;
                padding: 30px 0;
                border-bottom: 3px solid #007bff;
            }
            .report-header h1 {
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .report-meta {
                color: #6c757d;
                font-size: 0.9em;
            }
            .section {
                margin-bottom: 50px;
                padding: 20px;
                border-radius: 8px;
                background-color: #ffffff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #2c3e50;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .metric-label {
                font-size: 0.9em;
                color: #6c757d;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 1.4em;
                font-weight: bold;
                color: #2c3e50;
            }
            .plot-container {
                margin: 30px 0;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .data-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .data-table th, .data-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #dee2e6;
            }
            .data-table th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            .toc {
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 0;
            }
            .toc li {
                margin: 8px 0;
            }
            .toc a {
                color: #007bff;
                text-decoration: none;
            }
            .toc a:hover {
                text-decoration: underline;
            }
            .report-footer {
                text-align: center;
                margin-top: 50px;
                padding: 20px;
                border-top: 1px solid #dee2e6;
                color: #6c757d;
                font-size: 0.9em;
            }
        """
        
        if template_type == "executive":
            return base_styles + """
                .executive-summary {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 40px;
                }
                .executive-summary h2 {
                    color: white;
                    border-bottom: 2px solid rgba(255,255,255,0.3);
                }
                .key-insights {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .insight-card {
                    background-color: rgba(255,255,255,0.1);
                    padding: 20px;
                    border-radius: 8px;
                }
            """
        elif template_type == "technical":
            return base_styles + """
                .technical-details {
                    background-color: #f8f9fa;
                    border-left: 4px solid #28a745;
                    padding: 20px;
                    margin: 20px 0;
                }
                .code-block {
                    background-color: #2d3748;
                    color: #e2e8f0;
                    padding: 15px;
                    border-radius: 6px;
                    font-family: 'Courier New', monospace;
                    overflow-x: auto;
                }
                .warning-box {
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    color: #856404;
                    padding: 15px;
                    border-radius: 6px;
                    margin: 15px 0;
                }
            """
        elif template_type == "comparison":
            return base_styles + """
                .comparison-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                .strategy-card {
                    border: 2px solid #e9ecef;
                    border-radius: 10px;
                    padding: 20px;
                    transition: transform 0.2s;
                }
                .strategy-card:hover {
                    transform: translateY(-5px);
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }
                .performance-badge {
                    display: inline-block;
                    padding: 5px 10px;
                    border-radius: 15px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin: 5px;
                }
                .badge-excellent { background-color: #d4edda; color: #155724; }
                .badge-good { background-color: #cce7ff; color: #004085; }
                .badge-average { background-color: #fff3cd; color: #856404; }
                .badge-poor { background-color: #f8d7da; color: #721c24; }
            """
        
        return base_styles
    
    def _calculate_summary_statistics(self, results: Dict[str, VisualizationResult]) -> Dict[str, Any]:
        """Calculate summary statistics across all strategies."""
        summary = {
            'total_strategies': len(results),
            'successful_strategies': sum(1 for r in results.values() if r.success),
            'failed_strategies': sum(1 for r in results.values() if not r.success),
            'avg_generation_time': np.mean([r.generation_time for r in results.values()]),
            'total_generation_time': sum(r.generation_time for r in results.values()),
            'metrics_summary': {}
        }
        
        # Aggregate metrics across strategies
        all_metrics = {}
        for result in results.values():
            if result.success and result.metrics_summary:
                for metric, value in result.metrics_summary.items():
                    if isinstance(value, (int, float)):
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)
        
        # Calculate statistics for each metric
        for metric, values in all_metrics.items():
            if values:
                summary['metrics_summary'][metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return summary
    
    def _generate_executive_summary(self, summary_stats: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        return f"""
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="key-insights">
                <div class="insight-card">
                    <h3>Strategy Performance</h3>
                    <p>{summary_stats['successful_strategies']} out of {summary_stats['total_strategies']} strategies analyzed successfully</p>
                </div>
                <div class="insight-card">
                    <h3>Analysis Time</h3>
                    <p>Total processing time: {summary_stats['total_generation_time']:.2f} seconds</p>
                    <p>Average per strategy: {summary_stats['avg_generation_time']:.2f} seconds</p>
                </div>
                <div class="insight-card">
                    <h3>Key Metrics</h3>
                    <p>{len(summary_stats['metrics_summary'])} performance metrics calculated</p>
                    <p>Comprehensive risk and return analysis included</p>
                </div>
            </div>
        </section>
        """
    
    def _generate_table_of_contents(self, results: Dict[str, VisualizationResult], custom_sections: Optional[Dict[str, str]]) -> str:
        """Generate table of contents."""
        toc_items = ["<li><a href=\"#summary\">Summary Statistics</a></li>"]
        
        for name in results.keys():
            safe_name = name.replace(' ', '_').lower()
            toc_items.append(f"<li><a href=\"#{safe_name}\">{name}</a></li>")
        
        if len(results) > 1:
            toc_items.append("<li><a href=\"#comparison\">Comparative Analysis</a></li>")
        
        if custom_sections:
            for section_title in custom_sections.keys():
                safe_title = section_title.replace(' ', '_').lower()
                toc_items.append(f"<li><a href=\"#{safe_title}\">{section_title}</a></li>")
        
        return f"""
        <nav class="toc">
            <h2>Table of Contents</h2>
            <ul>
                {''.join(toc_items)}
            </ul>
        </nav>
        """
    
    def _generate_summary_section(self, summary_stats: Dict[str, Any]) -> str:
        """Generate summary statistics section."""
        metrics_cards = []
        
        for metric, stats in summary_stats['metrics_summary'].items():
            metrics_cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                <div class="metric-value">{stats['mean']:.4f}</div>
                <div class="metric-details">
                    <small>Range: {stats['min']:.4f} - {stats['max']:.4f}</small><br>
                    <small>Std Dev: {stats['std']:.4f}</small>
                </div>
            </div>
            """)
        
        return f"""
        <section id="summary" class="section">
            <h2>Summary Statistics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Strategies</div>
                    <div class="metric-value">{summary_stats['total_strategies']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Successful Analysis</div>
                    <div class="metric-value">{summary_stats['successful_strategies']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Generation Time</div>
                    <div class="metric-value">{summary_stats['avg_generation_time']:.2f}s</div>
                </div>
                {''.join(metrics_cards)}
            </div>
        </section>
        """
    
    def _generate_strategy_section(self, name: str, result: VisualizationResult, template_type: str, include_raw_data: bool) -> str:
        """Generate individual strategy section."""
        safe_name = name.replace(' ', '_').lower()
        
        # Generate metrics cards
        metrics_cards = []
        if result.metrics_summary:
            for metric, value in result.metrics_summary.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
                else:
                    formatted_value = str(value)
                
                metrics_cards.append(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric.replace('_', ' ').title()}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
                """)
        
        # Generate data table if requested
        data_table = ""
        if include_raw_data and result.plot_data:
            table_rows = []
            for series_name, series_data in result.plot_data.items():
                # Show first 10 and last 10 rows
                if len(series_data) > 20:
                    head_data = series_data.head(10)
                    tail_data = series_data.tail(10)
                    
                    for idx, value in head_data.items():
                        table_rows.append(f"<tr><td>{series_name}</td><td>{idx}</td><td>{value:.6f}</td></tr>")
                    
                    table_rows.append("<tr><td colspan='3'>... (data truncated) ...</td></tr>")
                    
                    for idx, value in tail_data.items():
                        table_rows.append(f"<tr><td>{series_name}</td><td>{idx}</td><td>{value:.6f}</td></tr>")
                else:
                    for idx, value in series_data.items():
                        table_rows.append(f"<tr><td>{series_name}</td><td>{idx}</td><td>{value:.6f}</td></tr>")
            
            data_table = f"""
            <h3>Raw Data Sample</h3>
            <table class="data-table">
                <thead>
                    <tr><th>Series</th><th>Index</th><th>Value</th></tr>
                </thead>
                <tbody>
                    {''.join(table_rows[:100])}  <!-- Limit to 100 rows -->
                </tbody>
            </table>
            """
        
        return f"""
        <section id="{safe_name}" class="section">
            <h2>{name}</h2>
            <div class="metrics-grid">
                {''.join(metrics_cards)}
            </div>
            <div class="plot-container">
                <h3>Visualization</h3>
                <p><em>Interactive plot available in separate HTML export: {name}_visualization.html</em></p>
                <p>Generation time: {result.generation_time:.2f} seconds</p>
            </div>
            {data_table}
        </section>
        """
    
    def _generate_comparative_section(self, results: Dict[str, VisualizationResult]) -> str:
        """Generate comparative analysis section."""
        # Create comparison table
        comparison_rows = []
        metrics_to_compare = set()
        
        # Collect all metrics
        for result in results.values():
            if result.success and result.metrics_summary:
                metrics_to_compare.update(result.metrics_summary.keys())
        
        # Generate comparison table
        for metric in sorted(metrics_to_compare):
            row = f"<tr><td>{metric.replace('_', ' ').title()}</td>"
            for name, result in results.items():
                if result.success and metric in result.metrics_summary:
                    value = result.metrics_summary[metric]
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
                    else:
                        formatted_value = str(value)
                    row += f"<td>{formatted_value}</td>"
                else:
                    row += "<td>N/A</td>"
            row += "</tr>"
            comparison_rows.append(row)
        
        strategy_headers = "".join([f"<th>{name}</th>" for name in results.keys()])
        
        return f"""
        <section id="comparison" class="section">
            <h2>Comparative Analysis</h2>
            <h3>Performance Comparison</h3>
            <table class="data-table">
                <thead>
                    <tr><th>Metric</th>{strategy_headers}</tr>
                </thead>
                <tbody>
                    {''.join(comparison_rows)}
                </tbody>
            </table>
            <div class="plot-container">
                <h3>Strategy Comparison Chart</h3>
                <p><em>Comparative visualization would be embedded here in full implementation</em></p>
            </div>
        </section>
        """
    
    def _generate_pdf_report(self, html_content: str, report_dir: Path, title: str) -> Optional[str]:
        """Generate PDF version of the report if possible."""
        try:
            # Try to import weasyprint for PDF generation
            try:
                import weasyprint
                pdf_path = report_dir / f"{title.replace(' ', '_')}_report.pdf"
                weasyprint.HTML(string=html_content).write_pdf(str(pdf_path))
                return str(pdf_path)
            except ImportError:
                self.logger.warning("weasyprint not available, skipping PDF generation")
                return None
        except Exception as e:
            self.logger.warning(f"PDF generation failed: {str(e)}")
            return None
    
    def _create_report_manifest(
        self,
        results: Dict[str, VisualizationResult],
        title: str,
        template_type: str,
        html_path: Path,
        pdf_path: Optional[str],
        exported_viz: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create report manifest with metadata."""
        return {
            'report_title': title,
            'template_type': template_type,
            'generation_timestamp': datetime.now().isoformat(),
            'html_report': str(html_path),
            'pdf_report': pdf_path,
            'strategies_analyzed': list(results.keys()),
            'successful_strategies': [name for name, result in results.items() if result.success],
            'failed_strategies': [name for name, result in results.items() if not result.success],
            'exported_visualizations': exported_viz,
            'total_generation_time': sum(r.generation_time for r in results.values()),
            'report_version': '1.0.0'
        }
    
    def _generate_comparative_report(
        self,
        strategy_results: Dict[str, Dict[str, VisualizationResult]],
        title: str,
        output_dir: Path
    ) -> str:
        """Generate comparative analysis report."""
        # Flatten results for comparison
        flattened_results = {}
        for strategy_name, results in strategy_results.items():
            for viz_type, result in results.items():
                flattened_results[f"{strategy_name}_{viz_type}"] = result
        
        report_path = output_dir / "comparative_analysis.html"
        html_content = self._generate_comprehensive_html_report(
            flattened_results,
            title,
            template_type="comparison"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_ranking_report(
        self,
        strategy_results: Dict[str, Dict[str, VisualizationResult]],
        output_dir: Path
    ) -> str:
        """Generate performance ranking report."""
        # Extract key metrics for ranking
        strategy_metrics = {}
        for strategy_name, results in strategy_results.items():
            metrics = {}
            for viz_type, result in results.items():
                if result.success and result.metrics_summary:
                    for metric, value in result.metrics_summary.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{viz_type}_{metric}"] = value
            strategy_metrics[strategy_name] = metrics
        
        # Generate ranking HTML
        ranking_html = self._generate_ranking_html(strategy_metrics)
        
        report_path = output_dir / "performance_ranking.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(ranking_html)
        
        return str(report_path)
    
    def _generate_risk_analysis_report(
        self,
        strategy_results: Dict[str, Dict[str, VisualizationResult]],
        output_dir: Path
    ) -> str:
        """Generate risk analysis report."""
        # Extract risk-related metrics
        risk_metrics = {}
        for strategy_name, results in strategy_results.items():
            risk_data = {}
            for viz_type, result in results.items():
                if result.success and result.metrics_summary:
                    # Look for risk-related metrics
                    for metric, value in result.metrics_summary.items():
                        if any(risk_term in metric.lower() for risk_term in ['drawdown', 'volatility', 'var', 'risk', 'sharpe']):
                            risk_data[f"{viz_type}_{metric}"] = value
            risk_metrics[strategy_name] = risk_data
        
        # Generate risk analysis HTML
        risk_html = self._generate_risk_html(risk_metrics)
        
        report_path = output_dir / "risk_analysis.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(risk_html)
        
        return str(report_path)
    
    def _generate_ranking_html(self, strategy_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate HTML for performance ranking."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Ranking Report</title>
            <style>{self._get_template_styles('comparison')}</style>
        </head>
        <body>
            <div class="report-container">
                <header class="report-header">
                    <h1>Performance Ranking Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                <section class="section">
                    <h2>Strategy Rankings</h2>
                    <p>Detailed performance rankings would be generated here based on key metrics.</p>
                    <div class="comparison-grid">
                        {''.join([f'<div class="strategy-card"><h3>{name}</h3><p>{len(metrics)} metrics analyzed</p></div>' 
                                for name, metrics in strategy_metrics.items()])}
                    </div>
                </section>
            </div>
        </body>
        </html>
        """
    
    def _generate_risk_html(self, risk_metrics: Dict[str, Dict[str, float]]) -> str:
        """Generate HTML for risk analysis."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Risk Analysis Report</title>
            <style>{self._get_template_styles('technical')}</style>
        </head>
        <body>
            <div class="report-container">
                <header class="report-header">
                    <h1>Risk Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                <section class="section">
                    <h2>Risk Assessment</h2>
                    <p>Comprehensive risk analysis across all strategies.</p>
                    <div class="technical-details">
                        <h3>Risk Metrics Summary</h3>
                        {''.join([f'<p><strong>{name}:</strong> {len(metrics)} risk metrics calculated</p>' 
                                for name, metrics in risk_metrics.items()])}
                    </div>
                </section>
            </div>
        </body>
        </html>
        """
    
    def cleanup_old_exports(self, days_old: int = 30) -> int:
        """
        Clean up old export files.
        
        Args:
            days_old: Remove files older than this many days
            
        Returns:
            Number of files removed
        """
        try:
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            removed_count = 0
            
            for file_path in self.export_dir.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1
            
            self.logger.info(f"Cleaned up {removed_count} old export files")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return 0