#!/usr/bin/env python3
"""
Complete test suite for Smart Document Processor
Tests all components systematically
"""

import requests
import json
import time
import sys
from pathlib import Path
import asyncio
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessorTester:
    """Complete testing suite for the document processor"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.test_results = []
        
    def test_health_check(self) -> bool:
        """Test system health endpoint"""
        print("🏥 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Version: {data.get('version', 'unknown')}")
                print(f"   Coordinator ready: {data.get('coordinator_ready', False)}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure it's running on port 8000")
            print("   Start with: python src/main.py")
            return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_agent_status(self) -> bool:
        """Test agent status endpoint"""
        print("📊 Testing agent status...")
        try:
            response = self.session.get(f"{self.base_url}/agent-status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Agent status retrieved successfully")
                
                # Print agent statuses
                print("\n🤖 Agent Statuses:")
                for agent_name, status in data.items():
                    if isinstance(status, dict) and "status" in status:
                        status_icon = "🟢" if status["status"] == "completed" else "🔴" if status["status"] == "error" else "🟡"
                        print(f"   {status_icon} {agent_name}: {status['status']}")
                        if "metrics" in status:
                            metrics = status["metrics"]
                            print(f"      Tasks: {metrics.get('total_tasks_processed', 0)} | Success: {metrics.get('success_rate', 0):.1%}")
                
                # Check system health
                system_health = data.get("system_health", {})
                if system_health.get("all_agents_healthy"):
                    print("✅ All agents healthy")
                else:
                    print("⚠️ Some agents have issues")
                
                return True
            else:
                print(f"❌ Agent status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Agent status error: {str(e)}")
            return False
    
    def test_config_endpoint(self) -> bool:
        """Test configuration endpoint"""
        print("⚙️ Testing configuration endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/config", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Configuration retrieved successfully")
                
                # Display key configuration
                if "agents" in data:
                    print(f"📋 Active Agents: {', '.join(data['agents'].keys())}")
                if "processing" in data:
                    processing = data["processing"]
                    print(f"📊 Supported formats: {', '.join(processing.get('supported_formats', []))}")
                    print(f"📏 Max file size: {processing.get('max_file_size_mb', 'unknown')} MB")
                
                return True
            else:
                print(f"❌ Configuration failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Configuration error: {str(e)}")
            return False
    
    def create_test_documents(self) -> List[Path]:
        """Create simple test documents if none exist"""
        test_dir = Path("test_documents")
        test_dir.mkdir(exist_ok=True)
        
        created_docs = []
        
        # Create a simple text-based PDF if PyPDF2 is available
        try:
            from fpdf import FPDF
            
            # Test Invoice
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="INVOICE #INV-2024-001", ln=1, align="C")
            pdf.cell(200, 10, txt="Date: January 15, 2024", ln=1)
            pdf.cell(200, 10, txt="Vendor: Tech Solutions Inc.", ln=1)
            pdf.cell(200, 10, txt="Total: $1,234.56", ln=1)
            pdf.cell(200, 10, txt="Tax: $123.45", ln=1)
            invoice_path = test_dir / "test_invoice.pdf"
            pdf.output(str(invoice_path))
            created_docs.append(invoice_path)
            print(f"✅ Created test invoice: {invoice_path}")
            
            # Test Contract
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="SERVICE AGREEMENT", ln=1, align="C")
            pdf.cell(200, 10, txt="This agreement is made on January 15, 2024", ln=1)
            pdf.cell(200, 10, txt="Between: Tech Solutions Inc. and Client Corp", ln=1)
            pdf.cell(200, 10, txt="Term: 12 months starting February 1, 2024", ln=1)
            contract_path = test_dir / "test_contract.pdf"
            pdf.output(str(contract_path))
            created_docs.append(contract_path)
            print(f"✅ Created test contract: {contract_path}")
            
        except ImportError:
            print("⚠️ fpdf not available, skipping PDF creation")
        
        # Create a simple text file as fallback
        text_path = test_dir / "test_document.txt"
        with open(text_path, "w") as f:
            f.write("SAMPLE INVOICE\n\n")
            f.write("Invoice #: INV-2024-001\n")
            f.write("Date: January 15, 2024\n")
            f.write("Company: Tech Solutions Inc.\n")
            f.write("Total: $1,234.56\n")
            f.write("Description: Software development services\n")
        created_docs.append(text_path)
        print(f"✅ Created test text document: {text_path}")
        
        return created_docs
    
    def find_test_documents(self) -> List[Path]:
        """Find existing test documents"""
        test_dir = Path("test_documents")
        
        if not test_dir.exists():
            print("⚠️ test_documents folder not found, creating...")
            return self.create_test_documents()
        
        # Find all supported document types
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.txt', '.docx']
        documents = []
        
        for ext in supported_extensions:
            documents.extend(test_dir.glob(f"*{ext}"))
        
        if not documents:
            print("⚠️ No test documents found, creating some...")
            return self.create_test_documents()
        
        print(f"✅ Found {len(documents)} test documents")
        for doc in documents:
            print(f"   📄 {doc.name}")
        
        return documents
    
    def test_document_processing(self, document_path: str, extraction_type: str = "text") -> bool:
        """Test document processing endpoint"""
        print(f"📄 Testing document processing: {Path(document_path).name}")
        
        try:
            with open(document_path, 'rb') as f:
                files = {'file': (Path(document_path).name, f, 'application/octet-stream')}
                data = {
                    'extraction_type': extraction_type,
                    'output_format': 'json',
                    'analysis_type': 'auto'
                }
                
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/process-document", 
                    files=files, 
                    data=data,
                    timeout=self.timeout
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Document processed successfully in {processing_time:.2f}s")
                    
                    # Print key results
                    doc_info = result.get('document_information', {})
                    analysis = result.get('analysis_results', {})
                    validation = result.get('validation_results', {})
                    metadata = result.get('metadata', {})
                    
                    print(f"   📊 Document: {doc_info.get('filename', 'Unknown')}")
                    print(f"   🏷️  Type: {analysis.get('document_type', 'Unknown')}")
                    print(f"   ✅ Valid: {validation.get('is_valid', False)}")
                    print(f"   📈 Confidence: {metadata.get('overall_confidence', 0):.2f}")
                    print(f"   ⏱️  Processing Time: {metadata.get('processing_time', 0):.2f}s")
                    
                    # Save results
                    output_file = f"test_results_{int(time.time())}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    
                    print(f"   💾 Results saved to: {output_file}")
                    
                    # Store test result
                    self.test_results.append({
                        "test": "document_processing",
                        "document": document_path,
                        "success": True,
                        "processing_time": processing_time,
                        "confidence": metadata.get('overall_confidence', 0),
                        "document_type": analysis.get('document_type', 'Unknown')
                    })
                    
                    return True
                    
                else:
                    print(f"❌ Document processing failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
                    self.test_results.append({
                        "test": "document_processing",
                        "document": document_path,
                        "success": False,
                        "error": response.text
                    })
                    
                    return False
                    
        except Exception as e:
            print(f"❌ Document processing error: {str(e)}")
            
            self.test_results.append({
                "test": "document_processing",
                "document": document_path,
                "success": False,
                "error": str(e)
            })
            
            return False
    
    def test_different_extraction_types(self, document_path: str) -> Dict[str, bool]:
        """Test different extraction types on the same document"""
        print(f"🔍 Testing different extraction types on: {Path(document_path).name}")
        
        extraction_types = ["text", "table", "structure", "full"]
        results = {}
        
        for ext_type in extraction_types:
            print(f"   Testing {ext_type} extraction...")
            success = self.test_document_processing(document_path, ext_type)
            results[ext_type] = success
            
            # Small delay between tests to avoid overwhelming the system
            time.sleep(1)
        
        return results
    
    def run_comprehensive_test(self, document_path: str) -> Dict[str, Any]:
        """Run comprehensive test on a single document"""
        print(f"🧪 Running comprehensive test on: {Path(document_path).name}")
        
        results = {
            "document": document_path,
            "basic_processing": False,
            "extraction_types": {},
            "performance": {},
            "quality_scores": []
        }
        
        # Basic processing test
        results["basic_processing"] = self.test_document_processing(document_path)
        
        # Different extraction types
        results["extraction_types"] = self.test_different_extraction_types(document_path)
        
        # Performance test (process same document multiple times)
        print("   📊 Running performance test...")
        processing_times = []
        for i in range(3):
            start = time.time()
            success = self.test_document_processing(document_path)
            if success:
                processing_times.append(time.time() - start)
            time.sleep(0.5)
        
        if processing_times:
            results["performance"] = {
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times)
            }
        
        return results
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.test_results:
            return "No tests were performed"
        
        successful_tests = sum(1 for result in self.test_results if result.get("success", False))
        total_tests = len(self.test_results)
        
        report = f"""
🧪 SMART DOCUMENT PROCESSOR - TEST REPORT
{"="*60}

📊 SUMMARY:
   Total Tests: {total_tests}
   Successful: {successful_tests}
   Failed: {total_tests - successful_tests}
   Success Rate: {successful_tests/total_tests:.1%}

📈 PERFORMANCE METRICS:
"""
        
        # Add performance metrics if available
        processing_times = [r.get("processing_time", 0) for r in self.test_results if r.get("success")]
        if processing_times:
            report += f"""
   Average Processing Time: {sum(processing_times)/len(processing_times):.2f}s
   Min Processing Time: {min(processing_times):.2f}s
   Max Processing Time: {max(processing_times):.2f}s
   
📋 CONFIDENCE SCORES:
"""
            confidence_scores = [r.get("confidence", 0) for r in self.test_results if r.get("success")]
            if confidence_scores:
                report += f"""
   Average Confidence: {sum(confidence_scores)/len(confidence_scores):.2f}
   Min Confidence: {min(confidence_scores):.2f}
   Max Confidence: {max(confidence_scores):.2f}
"""
        
        # Document types tested
        doc_types = list(set(r.get("document_type", "Unknown") for r in self.test_results if r.get("success")))
        if doc_types:
            report += f"\n📄 DOCUMENT TYPES TESTED:\n"
            for doc_type in doc_types:
                count = sum(1 for r in self.test_results if r.get("document_type") == doc_type and r.get("success"))
                report += f"   {doc_type}: {count} documents\n"
        
        report += "\n" + "="*60
        
        return report
    
    def run_full_test_suite(self) -> bool:
        """Run complete test suite"""
        print("🚀 Starting Smart Document Processor Test Suite")
        print("=" * 70)
        
        all_passed = True
        
        # Test 1: Health check
        if not self.test_health_check():
            all_passed = False
            print("❌ Health check failed - stopping tests")
            return False
        
        # Test 2: Agent status
        if not self.test_agent_status():
            all_passed = False
        
        # Test 3: Configuration
        if not self.test_config_endpoint():
            all_passed = False
        
        # Test 4: Find and test documents
        test_documents = self.find_test_documents()
        
        if test_documents:
            print(f"\n📁 Testing {len(test_documents)} documents...")
            
            for i, doc_path in enumerate(test_documents, 1):
                print(f"\n[{i}/{len(test_documents)}] Testing: {doc_path.name}")
                
                # Basic test
                if self.test_document_processing(str(doc_path)):
                    # Comprehensive test on first document
                    if i == 1:
                        comprehensive_results = self.run_comprehensive_test(str(doc_path))
                        print(f"   📊 Comprehensive test completed")
                else:
                    all_passed = False
                
                # Small delay between documents
                if i < len(test_documents):
                    time.sleep(2)
        else:
            print("❌ No test documents available")
            all_passed = False
        
        # Generate and display report
        print("\n" + "="*70)
        print(self.generate_test_report())
        
        # Save test results
        results_file = f"test_results_{int(time.time())}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "test_results": self.test_results,
                "summary": {
                    "total_tests": len(self.test_results),
                    "successful": sum(1 for r in self.test_results if r.get("success", False)),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return all_passed

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test Smart Document Processor")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    parser.add_argument("--document", help="Test specific document")
    parser.add_argument("--comprehensive", action="store_true", help="Run comprehensive tests")
    
    args = parser.parse_args()
    
    # Check if server is running
    print(f"🔍 Connecting to server at {args.server}...")
    try:
        response = requests.get(f"{args.server}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding properly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server")
        print("💡 Start the server first with: python src/main.py")
        return
    
    # Create tester and run tests
    tester = DocumentProcessorTester(args.server, args.timeout)
    
    if args.document:
        # Test specific document
        if args.comprehensive:
            results = tester.run_comprehensive_test(args.document)
            print(f"\n📊 Comprehensive test results: {results}")
        else:
            success = tester.test_document_processing(args.document)
            print(f"\n{'✅' if success else '❌'} Document test: {'PASSED' if success else 'FAILED'}")
    else:
        # Run full test suite
        success = tester.run_full_test_suite()
        print(f"\n{'🎉' if success else '⚠️'} Overall test result: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")

if __name__ == "__main__":
    import argparse
    asyncio.run(main())