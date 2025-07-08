import os
import logging
import tempfile
from typing import List, Optional, Union
from datetime import datetime
import shutil
import zipfile

# Setup logging
logger = logging.getLogger(__name__)

def export_slides_to_pdf(
    slide_files: List[str], 
    output_path: str,
    method: str = "auto",
    combine: bool = True
) -> dict:
    """
    Export HTML slides to PDF using multiple fallback methods
    
    Args:
        slide_files: List of HTML file paths to convert
        output_path: Output path for the PDF file(s)
        method: Export method ("playwright", "weasyprint", "auto")
        combine: Whether to combine all slides into one PDF
        
    Returns:
        dict with success status, output files, and any errors
    """
    try:
        logger.info(f"Starting PDF export with method: {method}")
        logger.info(f"Input slides: {len(slide_files)} files")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Combine slides: {combine}")
        
        if not slide_files:
            return {"success": False, "error": "No slide files provided"}
        
        # Validate slide files exist
        valid_files = []
        for file_path in slide_files:
            if os.path.exists(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Slide file not found: {file_path}")
        
        if not valid_files:
            return {"success": False, "error": "No valid slide files found"}
        
        logger.info(f"Found {len(valid_files)} valid slide files")
        
        # Try different export methods based on preference
        if method == "auto":
            # Only use playwright since weasyprint has Windows dependency issues
            methods_to_try = ["playwright"]
        else:
            methods_to_try = [method]
        
        last_error = None
        
        for export_method in methods_to_try:
            try:
                logger.info(f"Attempting PDF export with method: {export_method}")
                
                if export_method == "playwright":
                    result = _export_with_playwright(valid_files, output_path, combine)
                elif export_method == "weasyprint":
                    result = _export_with_weasyprint(valid_files, output_path, combine)
                else:
                    raise ValueError(f"Unknown export method: {export_method}")
                
                if result["success"]:
                    logger.info(f"PDF export successful with method: {export_method}")
                    result["method_used"] = export_method
                    return result
                else:
                    last_error = result.get("error", f"Unknown error with {export_method}")
                    logger.warning(f"Method {export_method} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error with method {export_method}: {last_error}")
                continue
        
        return {
            "success": False, 
            "error": f"All export methods failed. Last error: {last_error}"
        }
        
    except Exception as e:
        logger.error(f"Critical error in PDF export: {str(e)}")
        return {"success": False, "error": str(e)}

def _export_with_playwright(slide_files: List[str], output_path: str, combine: bool) -> dict:
    """Export slides to PDF using Playwright (browser-based rendering)"""
    try:
        # Import playwright in a way that's more compatible
        import subprocess
        import sys
        
        # First try direct import
        try:
            from playwright.sync_api import sync_playwright
            logger.info("Using Playwright for PDF export (direct import)")
            
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set page width to 1280px, let height be flexible
                page.set_viewport_size({"width": 1280, "height": 1024})  # Start with larger height
                
                individual_pdfs = []
                
                for i, slide_file in enumerate(slide_files):
                    try:
                        logger.info(f"Processing slide {i+1}/{len(slide_files)}: {os.path.basename(slide_file)}")
                        
                        # Load the HTML file
                        file_url = f"file://{os.path.abspath(slide_file)}"
                        page.goto(file_url, wait_until="networkidle")
                        
                        # Wait for fonts and styles to load
                        page.wait_for_timeout(3000)  # 3 second wait for better reliability
                        
                        # Get the actual slide container dimensions to fit exactly
                        slide_dimensions = page.evaluate("""() => {
                            const slideContainer = document.querySelector('.slide-container');
                            if (slideContainer) {
                                const rect = slideContainer.getBoundingClientRect();
                                const computedStyle = window.getComputedStyle(slideContainer);
                                return {
                                    width: Math.max(1280, rect.width, slideContainer.scrollWidth),
                                    height: Math.max(720, rect.height, slideContainer.scrollHeight)
                                };
                            } else {
                                // Fallback if no slide-container found
                                return {
                                    width: 1280,
                                    height: Math.max(
                                        document.body.scrollHeight,
                                        document.body.offsetHeight,
                                        720
                                    )
                                };
                            }
                        }""")
                        
                        content_width = slide_dimensions['width']
                        content_height = slide_dimensions['height']
                        
                        # Update viewport to match slide container dimensions
                        page.set_viewport_size({"width": content_width, "height": content_height})
                        
                        if combine:
                            # For combined PDF, save each slide as temporary PDF
                            temp_pdf = tempfile.NamedTemporaryFile(suffix=f"_slide_{i+1}.pdf", delete=False)
                            temp_pdf.close()
                            
                            # Use custom size based on slide container dimensions
                            page.pdf(
                                path=temp_pdf.name,
                                width=f"{content_width}px",
                                height=f"{content_height}px",
                                print_background=True,
                                margin={
                                    "top": "0px",
                                    "bottom": "0px", 
                                    "left": "0px",
                                    "right": "0px"
                                }
                            )
                            individual_pdfs.append(temp_pdf.name)
                            
                        else:
                            # Save individual PDF
                            slide_name = os.path.splitext(os.path.basename(slide_file))[0]
                            individual_output = output_path.replace(".pdf", f"_{slide_name}.pdf")
                            
                            # Use custom size based on slide container dimensions  
                            page.pdf(
                                path=individual_output,
                                width=f"{content_width}px",
                                height=f"{content_height}px",
                                print_background=True,
                                margin={
                                    "top": "0px",
                                    "bottom": "0px",
                                    "left": "0px", 
                                    "right": "0px"
                                }
                            )
                            individual_pdfs.append(individual_output)
                            
                    except Exception as e:
                        logger.error(f"Error processing slide {slide_file}: {str(e)}")
                        continue
                
                browser.close()
                
                if combine and len(individual_pdfs) > 1:
                    # Combine PDFs using PyPDF2
                    logger.info(f"Combining {len(individual_pdfs)} PDFs into one file")
                    success = _combine_pdfs(individual_pdfs, output_path)
                    
                    # Clean up temporary files
                    for temp_file in individual_pdfs:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    
                    if success:
                        return {
                            "success": True,
                            "output_files": [output_path],
                            "slides_exported": len(slide_files),
                            "method": "playwright"
                        }
                    else:
                        return {"success": False, "error": "Failed to combine PDFs"}
                        
                elif individual_pdfs:
                    return {
                        "success": True,
                        "output_files": individual_pdfs,
                        "slides_exported": len(individual_pdfs),
                        "method": "playwright"
                    }
                else:
                    return {"success": False, "error": "No PDFs were generated"}
                    
        except Exception as playwright_error:
            error_msg = str(playwright_error)
            logger.warning(f"Direct Playwright import failed: {error_msg}")
            
            # If there's a greenlet or compatibility issue, try subprocess approach
            if "greenlet" in error_msg.lower() or "module" in error_msg.lower():
                logger.info("Attempting subprocess-based PDF generation...")
                return _export_with_playwright_subprocess(slide_files, output_path, combine)
            else:
                raise playwright_error
                
    except ImportError:
        return {"success": False, "error": "Playwright not available"}
    except Exception as e:
        return {"success": False, "error": f"Playwright error: {str(e)}"}

def _export_with_playwright_subprocess(slide_files: List[str], output_path: str, combine: bool) -> dict:
    """Fallback method using subprocess to avoid greenlet issues"""
    try:
        import subprocess
        import json
        
        logger.info("Using Playwright subprocess method for PDF export")
        
        # Create a simple script that can be run in subprocess
        script_content = f'''
import os
import sys
from playwright.sync_api import sync_playwright

slide_files = {slide_files}
output_path = "{output_path}"
combine = {combine}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.set_viewport_size({{"width": 1280, "height": 1024}})
    
    individual_pdfs = []
    
    for i, slide_file in enumerate(slide_files):
        file_url = f"file://{{os.path.abspath(slide_file)}}"
        page.goto(file_url, wait_until="networkidle")
        page.wait_for_timeout(3000)
        
        # Get actual slide container dimensions
        slide_dimensions = page.evaluate("""() => {{
            const slideContainer = document.querySelector('.slide-container');
            if (slideContainer) {{
                const rect = slideContainer.getBoundingClientRect();
                return {{
                    width: Math.max(1280, rect.width, slideContainer.scrollWidth),
                    height: Math.max(720, rect.height, slideContainer.scrollHeight)
                }};
            }} else {{
                return {{
                    width: 1280,
                    height: Math.max(
                        document.body.scrollHeight,
                        document.body.offsetHeight,
                        720
                    )
                }};
            }}
        }}""")
        
        content_width = slide_dimensions['width']
        content_height = slide_dimensions['height']
        
        page.set_viewport_size({{"width": content_width, "height": content_height}})
        
        if combine:
            import tempfile
            temp_pdf = tempfile.NamedTemporaryFile(suffix=f"_slide_{{i+1}}.pdf", delete=False)
            temp_pdf.close()
            page.pdf(
                path=temp_pdf.name,
                width=f"{{content_width}}px",
                height=f"{{content_height}}px",
                print_background=True,
                margin={{"top": "0px", "bottom": "0px", "left": "0px", "right": "0px"}}
            )
            individual_pdfs.append(temp_pdf.name)
        else:
            slide_name = os.path.splitext(os.path.basename(slide_file))[0]
            individual_output = output_path.replace(".pdf", f"_{{slide_name}}.pdf")
            page.pdf(
                path=individual_output,
                width=f"{{content_width}}px",
                height=f"{{content_height}}px", 
                print_background=True,
                margin={{"top": "0px", "bottom": "0px", "left": "0px", "right": "0px"}}
            )
            individual_pdfs.append(individual_output)
    
    browser.close()
    
    # Output results
    print(json.dumps({{"individual_pdfs": individual_pdfs, "success": True}}))
'''
        
        # Write script to temporary file
        script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        script_file.write(script_content)
        script_file.close()
        
        try:
            # Run the script in subprocess
            result = subprocess.run([
                sys.executable, script_file.name
            ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
            
            if result.returncode == 0:
                # Parse the output
                output_data = json.loads(result.stdout.strip().split('\n')[-1])
                individual_pdfs = output_data['individual_pdfs']
                
                if combine and len(individual_pdfs) > 1:
                    # Combine PDFs
                    success = _combine_pdfs(individual_pdfs, output_path)
                    
                    # Clean up temporary files
                    for temp_file in individual_pdfs:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
                    
                    if success:
                        return {
                            "success": True,
                            "output_files": [output_path],
                            "slides_exported": len(slide_files),
                            "method": "playwright_subprocess"
                        }
                    else:
                        return {"success": False, "error": "Failed to combine PDFs"}
                        
                elif individual_pdfs:
                    return {
                        "success": True,
                        "output_files": individual_pdfs,
                        "slides_exported": len(individual_pdfs),
                        "method": "playwright_subprocess"
                    }
                else:
                    return {"success": False, "error": "No PDFs were generated"}
            else:
                return {"success": False, "error": f"Subprocess failed: {result.stderr}"}
                
        finally:
            # Clean up script file
            try:
                os.unlink(script_file.name)
            except:
                pass
                
    except Exception as e:
        return {"success": False, "error": f"Subprocess method failed: {str(e)}"}

def _export_with_weasyprint(slide_files: List[str], output_path: str, combine: bool) -> dict:
    """Export slides to PDF using WeasyPrint"""
    try:
        import weasyprint
        from weasyprint import HTML, CSS
        
        logger.info("Using WeasyPrint for PDF export")
        
        individual_pdfs = []
        
        for i, slide_file in enumerate(slide_files):
            try:
                logger.info(f"Processing slide {i+1}/{len(slide_files)}: {os.path.basename(slide_file)}")
                
                # Custom CSS for better PDF rendering
                custom_css = CSS(string="""
                    @page {
                        size: A4 landscape;
                        margin: 0;
                    }
                    body {
                        margin: 0;
                        padding: 0;
                    }
                    .slide-container {
                        width: 100vw;
                        height: 100vh;
                        page-break-after: always;
                    }
                """)
                
                html_doc = HTML(filename=slide_file)
                
                if combine:
                    # For combined PDF, save each slide as temporary PDF
                    temp_pdf = tempfile.NamedTemporaryFile(suffix=f"_slide_{i+1}.pdf", delete=False)
                    temp_pdf.close()
                    
                    html_doc.write_pdf(temp_pdf.name, stylesheets=[custom_css])
                    individual_pdfs.append(temp_pdf.name)
                    
                else:
                    # Save individual PDF  
                    slide_name = os.path.splitext(os.path.basename(slide_file))[0]
                    individual_output = output_path.replace(".pdf", f"_{slide_name}.pdf")
                    
                    html_doc.write_pdf(individual_output, stylesheets=[custom_css])
                    individual_pdfs.append(individual_output)
                    
            except Exception as e:
                logger.error(f"Error processing slide {slide_file}: {str(e)}")
                continue
        
        if combine and len(individual_pdfs) > 1:
            # Combine PDFs
            logger.info(f"Combining {len(individual_pdfs)} PDFs into one file")
            success = _combine_pdfs(individual_pdfs, output_path)
            
            # Clean up temporary files
            for temp_file in individual_pdfs:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            if success:
                return {
                    "success": True,
                    "output_files": [output_path],
                    "slides_exported": len(slide_files),
                    "method": "weasyprint"
                }
            else:
                return {"success": False, "error": "Failed to combine PDFs"}
                
        elif individual_pdfs:
            return {
                "success": True,
                "output_files": individual_pdfs,
                "slides_exported": len(individual_pdfs),
                "method": "weasyprint"
            }
        else:
            return {"success": False, "error": "No PDFs were generated"}
            
    except ImportError:
        return {"success": False, "error": "WeasyPrint not available"}
    except Exception as e:
        return {"success": False, "error": f"WeasyPrint error: {str(e)}"}

def _combine_pdfs(pdf_files: List[str], output_path: str) -> bool:
    """Combine multiple PDF files into one"""
    try:
        from PyPDF2 import PdfWriter, PdfReader
        
        pdf_writer = PdfWriter()
        
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        pdf_writer.add_page(page)
        
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
        
        logger.info(f"Successfully combined {len(pdf_files)} PDFs into {output_path}")
        return True
        
    except ImportError:
        logger.error("PyPDF2 not available for PDF combination")
        return False
    except Exception as e:
        logger.error(f"Error combining PDFs: {str(e)}")
        return False

def export_single_slide_to_pdf(slide_file: str, output_path: Optional[str] = None) -> dict:
    """
    Export a single HTML slide to PDF
    
    Args:
        slide_file: Path to the HTML slide file
        output_path: Optional output path for PDF. If None, will use slide filename.
        
    Returns:
        dict with success status and output information
    """
    if not output_path:
        slide_name = os.path.splitext(os.path.basename(slide_file))[0]
        output_dir = os.path.dirname(slide_file)
        output_path = os.path.join(output_dir, f"{slide_name}.pdf")
    
    result = export_slides_to_pdf([slide_file], output_path, combine=False)
    
    return result

def get_all_slides_in_directory(directory: str) -> List[str]:
    """Get all HTML slide files in a directory, sorted by name"""
    slide_files = []
    
    if os.path.exists(directory):
        for filename in sorted(os.listdir(directory)):
            if filename.endswith('.html') and filename.startswith('slide_'):
                slide_path = os.path.join(directory, filename)
                slide_files.append(slide_path)
    
    return slide_files

def export_all_slides_in_directory(directory: str, output_path: str) -> dict:
    """
    Export all slides in a directory to a combined PDF
    
    Args:
        directory: Directory containing HTML slide files
        output_path: Output path for the combined PDF
        
    Returns:
        dict with export results
    """
    slide_files = get_all_slides_in_directory(directory)
    
    if not slide_files:
        return {
            "success": False, 
            "error": f"No slide files found in directory: {directory}"
        }
    
    return export_slides_to_pdf(slide_files, output_path, combine=True) 