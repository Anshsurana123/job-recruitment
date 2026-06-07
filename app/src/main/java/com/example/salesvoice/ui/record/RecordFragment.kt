package com.example.salesvoice.ui.record

import android.Manifest
import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.example.salesvoice.data.model.SaleEntry
import com.example.salesvoice.databinding.FragmentRecordBinding
import com.example.salesvoice.ui.ViewModelFactory
import com.example.salesvoice.ui.settings.SettingsActivity
import com.example.salesvoice.voice.VoiceRecognitionManager
import com.google.android.material.snackbar.Snackbar
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.speech.SpeechRecognizer
import com.example.salesvoice.R

class RecordFragment : Fragment() {

    private var _binding: FragmentRecordBinding? = null
    private val binding get() = _binding!!

    private val viewModel: RecordViewModel by viewModels {
        ViewModelFactory(requireContext())
    }

    private lateinit var saleLogAdapter: SaleLogAdapter
    private var voiceManager: VoiceRecognitionManager? = null

    // Preferences
    private var currencySymbol = "₹"
    private var shopName = "My Shop"
    private var speechLanguage = "en-IN"

    // Animation variables
    private var pulseAnimatorSet: AnimatorSet? = null
    private val feedbackHideHandler = Handler(Looper.getMainLooper())
    private var hideRunnable: Runnable? = null

    // Permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startVoiceRecording()
        } else {
            showPermissionDeniedSnackbar()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentRecordBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        loadPreferences()
        setupHeader()
        setupVoiceManager()
        setupMicTouchListener()
        setupRecyclerView()
        observeViewModel()

        binding.btnSettings.setOnClickListener {
            startActivity(Intent(requireContext(), SettingsActivity::class.java))
        }
    }

    override fun onResume() {
        super.onResume()
        loadPreferences() // Reload if changed in Settings
        setupHeader()
        // Refresh adapters with new currency symbol if changed
        setupRecyclerView()
        checkOfflineLanguagePack()
    }

    private fun loadPreferences() {
        val sharedPrefs = requireActivity().getSharedPreferences("SalesVoicePrefs", Context.MODE_PRIVATE)
        currencySymbol = sharedPrefs.getString("pref_currency", "₹") ?: "₹"
        shopName = sharedPrefs.getString("pref_shop_name", "My Shop") ?: "My Shop"
        speechLanguage = sharedPrefs.getString("pref_language", "en-IN") ?: "en-IN"
    }

    private fun setupHeader() {
        val formatter = SimpleDateFormat("EEEE, d MMMM yyyy", Locale.getDefault())
        binding.tvTodayDate.text = formatter.format(Date())
    }

    private fun setupVoiceManager() {
        voiceManager = VoiceRecognitionManager(requireContext()).apply {
            onResult = { transcript ->
                activity?.runOnUiThread {
                    viewModel.processVoiceTranscript(transcript)
                }
            }
            onError = { errorCode ->
                activity?.runOnUiThread {
                    setUiState(MicState.DEFAULT)
                    if (errorCode == 13 || errorCode == 12) {
                        showOfflinePackDownloadDialog()
                    } else {
                        val errorMsg = when (errorCode) {
                            SpeechRecognizer.ERROR_NO_MATCH -> getString(R.string.voice_error_no_match)
                            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> getString(R.string.voice_error_timeout)
                            SpeechRecognizer.ERROR_AUDIO -> getString(R.string.voice_error_audio)
                            else -> getString(R.string.voice_error_generic, errorCode)
                        }
                        showFeedbackCard(SaleResult.Error(errorMsg))
                    }
                }
            }
            onOfflineMissing = {
                activity?.runOnUiThread {
                    setUiState(MicState.DEFAULT)
                    showOfflinePackDownloadDialog()
                }
            }
        }
    }

    private fun setupMicTouchListener() {
        binding.btnMic.setOnTouchListener { _, event ->
            // If processing in database, block new recording
            if (viewModel.isProcessing.value == true) return@setOnTouchListener false

            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    checkPermissionAndStart()
                    true
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    stopVoiceRecording()
                    true
                }
                else -> false
            }
        }
    }

    private fun checkPermissionAndStart() {
        val speechState = voiceManager?.getGoogleSpeechState() ?: VoiceRecognitionManager.GoogleSpeechState.OK
        if (speechState != VoiceRecognitionManager.GoogleSpeechState.OK) {
            showGoogleSpeechErrorDialog(speechState)
            return
        }

        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                startVoiceRecording()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.RECORD_AUDIO) -> {
                AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
                    .setTitle(getString(R.string.record_mic_permission_title))
                    .setMessage(getString(R.string.record_mic_permission_message))
                    .setPositiveButton(getString(R.string.record_mic_permission_allow)) { _, _ ->
                        requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                    .setNegativeButton(getString(R.string.cancel), null)
                    .show()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun showGoogleSpeechErrorDialog(state: VoiceRecognitionManager.GoogleSpeechState) {
        val builder = AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
        
        when (state) {
            VoiceRecognitionManager.GoogleSpeechState.DISABLED -> {
                val disabledPkg = voiceManager?.getDisabledPackageName() ?: "com.google.android.googlequicksearchbox"
                val appName = if (disabledPkg == "com.google.android.tts") "Speech Recognition & Synthesis" else "Google App"
                builder.setTitle("$appName is Disabled")
                    .setMessage("$appName is currently disabled on your phone. Please enable it to use offline voice commands.")
                    .setPositiveButton("Enable in Settings") { _, _ ->
                        try {
                            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                                data = Uri.fromParts("package", disabledPkg, null)
                            }
                            startActivity(intent)
                        } catch (e: Exception) {
                            try {
                                startActivity(Intent(Settings.ACTION_SETTINGS))
                            } catch (ex: Exception) {
                                Toast.makeText(context, "Could not open settings", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
            }
            VoiceRecognitionManager.GoogleSpeechState.NOT_INSTALLED -> {
                builder.setTitle("Google Speech Services Missing")
                    .setMessage("Google Speech Services is not installed on this device. Please install it from the Play Store to use offline voice commands.")
                    .setPositiveButton("Open Play Store") { _, _ ->
                        try {
                            startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=com.google.android.tts")))
                        } catch (e: Exception) {
                            try {
                                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://play.google.com/store/apps/details?id=com.google.android.tts")))
                            } catch (ex: Exception) {
                                try {
                                    startActivity(Intent(Settings.ACTION_SETTINGS))
                                } catch (ex2: Exception) {
                                    Toast.makeText(context, "Could not open Play Store", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    }
            }
            VoiceRecognitionManager.GoogleSpeechState.SERVICE_UNAVAILABLE -> {
                builder.setTitle("Google Voice Typing Off")
                    .setMessage("Google Voice Typing is currently turned off or restricted in your keyboard settings. Please enable it under 'Manage Keyboards' to use offline voice commands.")
                    .setPositiveButton("Turn On Keyboards") { _, _ ->
                        try {
                            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
                        } catch (e: Exception) {
                            try {
                                startActivity(Intent(Settings.ACTION_SETTINGS))
                            } catch (ex: Exception) {
                                Toast.makeText(context, "Could not open keyboard settings", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
            }
            else -> {}
        }
        
        builder.setNegativeButton(getString(R.string.cancel), null).show()
    }

    private fun startVoiceRecording() {
        setUiState(MicState.RECORDING)
        voiceManager?.startListening(speechLanguage)
    }

    private fun stopVoiceRecording() {
        if (binding.tvMicStatusLabel.text == getString(R.string.record_listening)) {
            setUiState(MicState.PROCESSING)
            voiceManager?.stopListening()
        }
    }

    private fun showOfflinePackDownloadDialog() {
        AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
            .setTitle(getString(R.string.record_offline_pack_title))
            .setMessage(getString(R.string.record_offline_pack_message))
            .setPositiveButton(getString(R.string.record_open_settings)) { _, _ ->
                try {
                    val intent = Intent(Settings.ACTION_LOCALE_SETTINGS)
                    startActivity(intent)
                } catch (e: Exception) {
                    try {
                        val intent = Intent(Settings.ACTION_SETTINGS)
                        startActivity(intent)
                    } catch (ex: Exception) {
                        Toast.makeText(context, "Could not open settings", Toast.LENGTH_SHORT).show()
                    }
                }
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    private fun showPermissionDeniedSnackbar() {
        Snackbar.make(
            binding.root,
            getString(R.string.record_mic_permission_snackbar),
            Snackbar.LENGTH_LONG
        ).setAction(getString(R.string.settings)) {
            val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS).apply {
                data = Uri.fromParts("package", requireContext().packageName, null)
            }
            startActivity(intent)
        }.show()
    }

    private fun setupRecyclerView() {
        saleLogAdapter = SaleLogAdapter(currencySymbol) { sale ->
            showDeleteConfirmation(sale)
        }
        binding.rvSalesLog.adapter = saleLogAdapter
    }

    private fun showDeleteConfirmation(sale: SaleEntry) {
        AlertDialog.Builder(requireContext(), com.google.android.material.R.style.Theme_MaterialComponents_Light_Dialog_Alert)
            .setTitle(getString(R.string.record_delete_title))
            .setMessage(getString(R.string.record_delete_message, sale.quantity.toString(), sale.unit, sale.productName))
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                viewModel.deleteSale(sale.id)
                Toast.makeText(context, getString(R.string.record_sale_deleted), Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    private fun observeViewModel() {
        viewModel.todaySales.observe(viewLifecycleOwner) { sales ->
            if (sales.isNullOrEmpty()) {
                binding.tvEmptyLogState.visibility = View.VISIBLE
                binding.rvSalesLog.visibility = View.GONE
                binding.badgeSalesCount.text = getString(R.string.record_sales_zero)
            } else {
                binding.tvEmptyLogState.visibility = View.GONE
                binding.rvSalesLog.visibility = View.VISIBLE
                saleLogAdapter.submitList(sales)
                binding.badgeSalesCount.text = getString(R.string.record_sales_count, sales.size)
            }
        }

        viewModel.todayTotals.observe(viewLifecycleOwner) { totals ->
            val rev = totals?.totalRevenue ?: 0.0
            val prof = totals?.totalProfit ?: 0.0
            binding.tvTotalRevenue.text = getString(R.string.record_total_format, "$currencySymbol${String.format(Locale.getDefault(), "%.2f", rev)}")
            binding.tvTotalProfit.text = getString(R.string.record_profit_format, "$currencySymbol${String.format(Locale.getDefault(), "%.2f", prof)}")
        }

        viewModel.allProductsCount.observe(viewLifecycleOwner) { count ->
            if (count == 0) {
                binding.tvWarningBanner.visibility = View.VISIBLE
            } else {
                binding.tvWarningBanner.visibility = View.GONE
            }
        }

        viewModel.isProcessing.observe(viewLifecycleOwner) { isProcessing ->
            if (isProcessing) {
                setUiState(MicState.PROCESSING)
            }
        }

        viewModel.saleResult.observe(viewLifecycleOwner) { result ->
            if (result != null) {
                showFeedbackCard(result)
                setUiState(MicState.DEFAULT)
                viewModel.clearSaleResult()
            }
        }
    }

    private enum class MicState { DEFAULT, RECORDING, PROCESSING }

    private fun setUiState(state: MicState) {
        when (state) {
            MicState.DEFAULT -> {
                binding.btnMic.isEnabled = true
                binding.btnMic.setImageResource(com.example.salesvoice.R.drawable.ic_mic)
                binding.btnMic.setBackgroundResource(com.example.salesvoice.R.drawable.circle_background)
                binding.tvMicStatusLabel.text = getString(R.string.record_hold_to_speak)
                binding.pbProcessing.visibility = View.GONE
                stopPulseAnimation()
            }
            MicState.RECORDING -> {
                binding.btnMic.setImageResource(com.example.salesvoice.R.drawable.ic_mic)
                binding.btnMic.setBackgroundResource(com.example.salesvoice.R.drawable.circle_background_recording)
                binding.tvMicStatusLabel.text = getString(R.string.record_listening)
                binding.pbProcessing.visibility = View.GONE
                startPulseAnimation()
            }
            MicState.PROCESSING -> {
                binding.btnMic.isEnabled = false
                binding.btnMic.setImageResource(0) // Hide mic icon to show spinner clearly
                binding.tvMicStatusLabel.text = getString(R.string.record_processing)
                binding.pbProcessing.visibility = View.VISIBLE
                stopPulseAnimation()
            }
        }
    }

    private fun startPulseAnimation() {
        binding.viewPulse1.visibility = View.VISIBLE
        binding.viewPulse2.visibility = View.VISIBLE

        val pulse1X = ObjectAnimator.ofFloat(binding.viewPulse1, "scaleX", 1f, 1.5f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
        }
        val pulse1Y = ObjectAnimator.ofFloat(binding.viewPulse1, "scaleY", 1f, 1.5f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
        }
        val alpha1 = ObjectAnimator.ofFloat(binding.viewPulse1, "alpha", 1f, 0f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
        }

        val pulse2X = ObjectAnimator.ofFloat(binding.viewPulse2, "scaleX", 1f, 1.5f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
            startDelay = 500
        }
        val pulse2Y = ObjectAnimator.ofFloat(binding.viewPulse2, "scaleY", 1f, 1.5f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
            startDelay = 500
        }
        val alpha2 = ObjectAnimator.ofFloat(binding.viewPulse2, "alpha", 1f, 0f).apply {
            repeatCount = ObjectAnimator.INFINITE
            repeatMode = ObjectAnimator.RESTART
            startDelay = 500
        }

        pulseAnimatorSet = AnimatorSet().apply {
            playTogether(pulse1X, pulse1Y, alpha1, pulse2X, pulse2Y, alpha2)
            duration = 1000
            start()
        }
    }

    private fun stopPulseAnimation() {
        pulseAnimatorSet?.cancel()
        pulseAnimatorSet = null
        binding.viewPulse1.visibility = View.INVISIBLE
        binding.viewPulse2.visibility = View.INVISIBLE
    }

    private fun showFeedbackCard(result: SaleResult) {
        // Cancel any pending hide timers
        hideRunnable?.let { feedbackHideHandler.removeCallbacks(it) }

        // Setup Card colors & details
        when (result) {
            is SaleResult.Success -> {
                binding.cvFeedbackCard.setCardBackgroundColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.success_light))
                binding.tvFeedbackTitle.text = getString(R.string.record_sale_logged)
                binding.tvFeedbackTitle.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.success))
                val entry = result.entry
                binding.tvFeedbackDetail.text = getString(
                    R.string.record_sale_detail,
                    entry.productName,
                    entry.quantity.toString(),
                    entry.unit,
                    "$currencySymbol${String.format(Locale.getDefault(), "%.2f", entry.totalAmount)}",
                    "$currencySymbol${String.format(Locale.getDefault(), "%.2f", entry.totalProfit)}"
                )
                binding.tvFeedbackDetail.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.text_primary))
            }
            is SaleResult.ProductNotFound -> {
                binding.cvFeedbackCard.setCardBackgroundColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.error_light))
                binding.tvFeedbackTitle.text = getString(R.string.record_product_not_found)
                binding.tvFeedbackTitle.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.error))
                binding.tvFeedbackDetail.text = getString(R.string.record_product_not_found_detail, result.productNameSpoken)
                binding.tvFeedbackDetail.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.text_primary))
            }
            is SaleResult.Error -> {
                binding.cvFeedbackCard.setCardBackgroundColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.error_light))
                binding.tvFeedbackTitle.text = getString(R.string.record_sale_failed)
                binding.tvFeedbackTitle.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.error))
                val displayMsg = if (result.message.contains("Could not find a quantity") || 
                                     result.message.contains("Couldn't extract quantity") ||
                                     result.message.contains("Could not find product name")) {
                    getString(R.string.parser_error_no_quantity)
                } else {
                    result.message
                }
                binding.tvFeedbackDetail.text = displayMsg
                binding.tvFeedbackDetail.setTextColor(ContextCompat.getColor(requireContext(), com.example.salesvoice.R.color.text_primary))
            }
        }

        // Animate Slide In
        binding.cvFeedbackCard.translationY = -100f
        binding.cvFeedbackCard.alpha = 0f
        binding.cvFeedbackCard.visibility = View.VISIBLE
        binding.cvFeedbackCard.animate()
            .translationY(0f)
            .alpha(1f)
            .setDuration(300)
            .setListener(null)

        // Set Hide Timer (4 seconds)
        hideRunnable = Runnable {
            binding.cvFeedbackCard.animate()
                .translationY(-100f)
                .alpha(0f)
                .setDuration(300)
                .setListener(object : AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: Animator) {
                        binding.cvFeedbackCard.visibility = View.GONE
                    }
                })
        }
        hideRunnable?.let { feedbackHideHandler.postDelayed(it, 4000) }
    }

    private fun checkOfflineLanguagePack() {
        voiceManager?.checkAndDownloadLanguagePack(speechLanguage) {
            activity?.runOnUiThread {
                Toast.makeText(
                    requireContext(),
                    getString(R.string.record_offline_download_started),
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        hideRunnable?.let { feedbackHideHandler.removeCallbacks(it) }
        voiceManager?.destroy()
        _binding = null
    }
}
