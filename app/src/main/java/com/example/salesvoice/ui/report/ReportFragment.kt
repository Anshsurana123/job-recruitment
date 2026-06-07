package com.example.salesvoice.ui.report

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import android.widget.Toast
import com.example.salesvoice.R
import com.example.salesvoice.databinding.FragmentReportBinding
import com.example.salesvoice.ui.ViewModelFactory
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ReportFragment : Fragment() {

    private var _binding: FragmentReportBinding? = null
    private val binding get() = _binding!!

    private val viewModel: ReportViewModel by viewModels {
        ViewModelFactory(requireContext())
    }

    private lateinit var reportAdapter: ReportAdapter
    private var currencySymbol = "₹"
    private var shopName = "My Shop"

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentReportBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        loadPreferences()
        setupRecyclerView()
        setupDateNavigation()
        observeViewModel()
        setupShareButton()
    }

    override fun onResume() {
        super.onResume()
        loadPreferences()
        setupRecyclerView() // Refresh with potential new currency symbol
    }

    private fun loadPreferences() {
        val sharedPrefs = requireActivity().getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)
        currencySymbol = sharedPrefs.getString("pref_currency", "₹") ?: "₹"
        shopName = sharedPrefs.getString("pref_shop_name", "My Shop") ?: "My Shop"
    }

    private fun setupRecyclerView() {
        reportAdapter = ReportAdapter(currencySymbol)
        binding.rvReportBreakdown.adapter = reportAdapter
    }

    private fun setupDateNavigation() {
        binding.btnPrevDay.setOnClickListener {
            viewModel.moveToPreviousDay()
        }
        binding.btnNextDay.setOnClickListener {
            viewModel.moveToNextDay()
        }
    }

    private fun observeViewModel() {
        // Observe Current Date Object for Title Display
        val uiFormatter = SimpleDateFormat("EEEE, d MMMM yyyy", Locale.getDefault())
        viewModel.currentDate.observe(viewLifecycleOwner) { date ->
            binding.tvSelectedDate.text = uiFormatter.format(date)
        }

        // Enable/Disable forward day navigation
        val dbSdf = SimpleDateFormat("yyyy-MM-dd", Locale.US)
        viewModel.selectedDateStr.observe(viewLifecycleOwner) { currentDbStr ->
            val todayDbStr = dbSdf.format(Date())
            binding.btnNextDay.isEnabled = currentDbStr != todayDbStr
            binding.btnNextDay.alpha = if (currentDbStr == todayDbStr) 0.3f else 1.0f
        }

        // Observe aggregated totals
        viewModel.totalsForDate.observe(viewLifecycleOwner) { totals ->
            val rev = totals?.totalRevenue ?: 0.0
            val prof = totals?.totalProfit ?: 0.0
            binding.tvReportRevenueTotal.text = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", rev)}"
            binding.tvReportProfitTotal.text = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", prof)}"
        }

        // Observe sales entries count
        viewModel.salesForDate.observe(viewLifecycleOwner) { sales ->
            val count = sales?.size ?: 0
            binding.tvReportSalesCountTotal.text = if (count == 1) getString(R.string.report_one_entry) else getString(R.string.report_entries, count)
        }

        // Observe product-wise table breakdown
        viewModel.productBreakdowns.observe(viewLifecycleOwner) { breakdownList ->
            if (breakdownList.isNullOrEmpty()) {
                binding.tvReportEmptyState.visibility = View.VISIBLE
                binding.rvReportBreakdown.visibility = View.GONE
                binding.llTableHeader.visibility = View.GONE
                binding.viewDivider.visibility = View.GONE
            } else {
                binding.tvReportEmptyState.visibility = View.GONE
                binding.rvReportBreakdown.visibility = View.VISIBLE
                binding.llTableHeader.visibility = View.VISIBLE
                binding.viewDivider.visibility = View.VISIBLE
                reportAdapter.submitList(breakdownList)
            }
        }
    }

    private fun setupShareButton() {
        binding.btnShareReport.setOnClickListener {
            val breakdowns = viewModel.productBreakdowns.value ?: emptyList()
            if (breakdowns.isEmpty()) {
                Toast.makeText(context, getString(R.string.report_no_sales_share), Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val dateStr = binding.tvSelectedDate.text.toString()
            val totalRevenue = binding.tvReportRevenueTotal.text.toString()
            val totalProfit = binding.tvReportProfitTotal.text.toString()
            val totalSales = viewModel.salesForDate.value?.size ?: 0

            // Generate report text
            val reportBuilder = StringBuilder().apply {
                append(getString(R.string.report_share_header, dateStr) + "\n")
                append(getString(R.string.report_share_shop, shopName) + "\n")
                append(getString(R.string.report_share_separator) + "\n")
                breakdowns.forEach { b ->
                    val qtyText = "${b.totalQuantity} ${b.unit}"
                    val revText = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", b.totalRevenue)}"
                    val profText = "$currencySymbol${String.format(Locale.getDefault(), "%.2f", b.totalProfit)}"
                    append("${b.productName}: $qtyText → $revText (${getString(R.string.sale_profit_format, profText)})\n")
                }
                append(getString(R.string.report_share_separator) + "\n")
                append(getString(R.string.report_share_total_revenue, totalRevenue) + "\n")
                append(getString(R.string.report_share_total_profit, totalProfit) + "\n")
                append(getString(R.string.report_share_total_sales, totalSales.toString()) + "\n")
            }

            // Launch standard Android share sheet
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, getString(R.string.report_share_subject, dateStr))
                putExtra(Intent.EXTRA_TEXT, reportBuilder.toString())
            }
            startActivity(Intent.createChooser(shareIntent, getString(R.string.report_share_chooser)))
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
